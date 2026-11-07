from collections.abc import Sequence
from unittest.mock import patch

import httpx
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from inspect_evals_utils.huggingface import (
    BACKOFF_CONFIG,
    CI_BACKOFF_CONFIG,
    STANDARD_BACKOFF_CONFIG,
    get_backoff_config,
    hf_backoff_policy,
    hf_dataset,
    hf_hub_download,
    load_dataset,
    set_backoff_config,
    set_backoff_recorder,
    set_hf_call_recorder,
    snapshot_download,
)


def _httpx_read_timeout(url: str) -> httpx.ReadTimeout:
    """Build an httpx.ReadTimeout with an attached Request, as real downloads raise."""
    err = httpx.ReadTimeout("The read operation timed out")
    err.request = httpx.Request("GET", url)
    return err


def _httpx_status_error(status_code: int, url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", url)
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


HF_FUNCTIONS = {
    "hf_dataset": hf_dataset,
    "snapshot_download": snapshot_download,
    "load_dataset": load_dataset,
    "hf_hub_download": hf_hub_download,
}


def hf_call_simulated(responses: Sequence[Exception | None] | None):
    def inner(*args, **kwargs):
        if not responses:
            return {"result": "ok"}

        result = responses.pop(0)
        if result is None:
            return {"result": "ok"}
        raise result

    return inner


def test_backoff_waits_and_retries():
    responses = [
        FileNotFoundError(
            "An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on."
        ),
        LocalEntryNotFoundError("Error"),
        None,
    ]
    sleep_calls = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append

        wrapped = hf_backoff_policy(
            **BACKOFF_CONFIG,
        )(hf_call_simulated(responses))

        result = wrapped("foo")

    assert result == {"result": "ok"}
    assert len(sleep_calls) == 2


def test_backoff_raises():
    max_tries = BACKOFF_CONFIG["max_tries"]
    responses = [LocalEntryNotFoundError("Error")] * (max_tries + 1)
    sleep_calls = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append

        wrapped = hf_backoff_policy(
            **BACKOFF_CONFIG,
        )(hf_call_simulated(responses))

        with pytest.raises(LocalEntryNotFoundError):
            wrapped("foo")


class NonTransientError(Exception):
    pass


def test_raises_on_non_transient():
    responses = [NonTransientError()]
    sleep_calls = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        with pytest.raises(NonTransientError):
            wrapped("foo")
        mock_sleep.assert_not_called()


def test_backoff_retries_on_httpx_read_timeout_for_hf() -> None:
    """httpx.ReadTimeout against huggingface.co should retry.

    Regression: hub 1.x migrated from `requests` to `httpx`, so timeouts no
    longer raise `requests.exceptions.ReadTimeout`.
    """
    responses: list[Exception | None] = [
        _httpx_read_timeout(
            "https://huggingface.co/datasets/openai/gdpval/resolve/main/data.parquet"
        ),
        None,
    ]
    sleep_calls: list[float] = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        result = wrapped("foo")

    assert result == {"result": "ok"}
    assert len(sleep_calls) == 1


def test_backoff_gives_up_on_httpx_read_timeout_for_non_hf_url() -> None:
    """A non-HF httpx timeout should not be retried.

    Same conservatism as the `requests.exceptions.ReadTimeout` branch.
    """
    responses: list[Exception | None] = [
        _httpx_read_timeout("https://example.com/foo"),
    ]
    sleep_calls: list[float] = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        with pytest.raises(httpx.ReadTimeout):
            wrapped("foo")

    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc/def?X-Amz-Signature=xyz",
        "https://cdn-lfs.hf.co/repos/foo/bar.parquet",
        "https://cdn-lfs-us-1.hf.co/repos/foo/bar.parquet",
    ],
)
def test_backoff_retries_on_httpx_read_timeout_for_hf_cdn(url: str) -> None:
    """Large file downloads redirect from huggingface.co to hf.co CDN hosts.

    A timeout raised mid-stream carries the post-redirect URL on the CDN, so
    the URL match must cover the alternate hf.co apex, not only huggingface.co.
    """
    responses: list[Exception | None] = [
        _httpx_read_timeout(url),
        None,
    ]
    sleep_calls: list[float] = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        result = wrapped("foo")

    assert result == {"result": "ok"}
    assert len(sleep_calls) == 1


@pytest.mark.parametrize("status_code", [429, 502, 503, 504])
def test_backoff_retries_on_httpx_status_error_transient(status_code: int) -> None:
    """httpx.HTTPStatusError with a transient code (429/502/503/504) should retry."""
    responses: list[Exception | None] = [
        _httpx_status_error(status_code, "https://huggingface.co/api/datasets/foo"),
        None,
    ]
    sleep_calls: list[float] = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        result = wrapped("foo")

    assert result == {"result": "ok"}
    assert len(sleep_calls) == 1


def test_backoff_gives_up_on_httpx_status_error_non_transient() -> None:
    """A non-transient httpx status error (e.g. 404) should not be retried."""
    responses: list[Exception | None] = [
        _httpx_status_error(404, "https://huggingface.co/api/datasets/missing"),
    ]
    sleep_calls: list[float] = []

    with patch("time.sleep") as mock_sleep:
        mock_sleep.side_effect = sleep_calls.append
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))

        with pytest.raises(httpx.HTTPStatusError):
            wrapped("foo")

    mock_sleep.assert_not_called()


def test_hf_call_recorder_receives_wrapped_function_names() -> None:
    calls: list[str] = []
    set_hf_call_recorder(calls.append)
    try:
        with patch("inspect_ai.dataset.hf_dataset", return_value={"result": "ok"}):
            result: object = hf_dataset(path="org/repo", split="train", revision="123")
    finally:
        set_hf_call_recorder(None)

    assert result == {"result": "ok"}
    assert calls == ["hf_dataset"]

    with patch("inspect_ai.dataset.hf_dataset", return_value={"result": "ok"}):
        hf_dataset(path="org/repo", split="train", revision="123")
    assert calls == ["hf_dataset"], "a removed recorder must not be called"


def test_hf_dataset_disables_inspect_ai_retry() -> None:
    with patch("inspect_ai.dataset.hf_dataset", return_value={"result": "ok"}) as mock:
        result: object = hf_dataset(path="org/repo", split="train", revision="123", retry=True)

    assert result == {"result": "ok"}
    mock.assert_called_once_with(path="org/repo", split="train", revision="123", retry=False)


def test_backoff_recorder_receives_backoff_details() -> None:
    details: list[dict] = []
    set_backoff_recorder(details.append)
    try:
        with patch("time.sleep"):
            wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(
                hf_call_simulated([LocalEntryNotFoundError("Error"), None])
            )
            wrapped("foo")
    finally:
        set_backoff_recorder(None)

    assert len(details) == 1
    assert details[0]["tries"] == 1
    assert details[0]["wait"] > 0


def test_set_backoff_config_applies_to_the_wrappers() -> None:
    set_backoff_config({**BACKOFF_CONFIG, "max_tries": 1})
    try:
        with (
            patch("time.sleep") as mock_sleep,
            patch("datasets.load_dataset", side_effect=LocalEntryNotFoundError("Error")),
            pytest.raises(LocalEntryNotFoundError),
        ):
            load_dataset("org/repo", revision="123")
        mock_sleep.assert_not_called()
    finally:
        set_backoff_config(None)
    assert get_backoff_config() == BACKOFF_CONFIG


def test_default_backoff_config_follows_the_ci_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CI", "1")
    set_backoff_config(None)
    assert get_backoff_config() == CI_BACKOFF_CONFIG
    monkeypatch.delenv("CI")
    set_backoff_config(None)
    assert get_backoff_config() == STANDARD_BACKOFF_CONFIG


@pytest.fixture(autouse=True)
def online_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)


@pytest.mark.parametrize(
    "error",
    [
        LocalEntryNotFoundError("Outgoing traffic has been disabled"),
        ConnectionError("Couldn't reach 'org/repo' on the Hub"),
    ],
)
def test_backoff_gives_up_in_offline_mode(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    responses: list[Exception | None] = [error]

    with patch("time.sleep") as mock_sleep:
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))
        with pytest.raises(type(error)):
            wrapped("foo")

    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "Couldn't reach 'org/repo' on the Hub (OfflineModeIsEnabled)",
        "Unable to connect to another service",
    ],
)
def test_backoff_gives_up_on_non_transient_connection_error(message: str) -> None:
    responses: list[Exception | None] = [ConnectionError(message)]

    with patch("time.sleep") as mock_sleep:
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))
        with pytest.raises(ConnectionError):
            wrapped("foo")

    mock_sleep.assert_not_called()


def test_backoff_retries_on_hub_connection_error() -> None:
    """Datasets can discard the original exception when the Hub is unreachable."""
    error = ConnectionError("Couldn't reach 'org/repo' on the Hub (LocalEntryNotFoundError)")
    responses: list[Exception | None] = [error, None]

    with patch("time.sleep") as mock_sleep:
        wrapped = hf_backoff_policy(**BACKOFF_CONFIG)(hf_call_simulated(responses))
        assert wrapped("foo") == {"result": "ok"}

    mock_sleep.assert_called_once()
