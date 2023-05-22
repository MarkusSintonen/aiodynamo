import asyncio
import datetime
import json
import time
import uuid
from contextlib import contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
)
from unittest.mock import mock_open, patch

import aiohttp
import pytest
from _pytest.monkeypatch import MonkeyPatch
from aiohttp import ClientSession, web
from yarl import URL

from aiodynamo.credentials import (
    AssumeRoleWebIdentityClientException,
    AssumeRoleWebIdentityHTTPStatusException,
    AssumeRoleWithWebIdentityEnvironmentCredentials,
    Key,
    Metadata,
    TooManyRetries,
)
from aiodynamo.http.aiohttp import AIOHTTP
from aiodynamo.http.types import HttpImplementation
from aiodynamo.models import RetryConfig
from aiodynamo.types import Seconds


def sts_mock_response(role_arn: str, metadata: Metadata) -> Dict[str, Any]:
    # Response format from https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
    response = {
        "AssumeRoleWithWebIdentityResponse": {
            "AssumeRoleWithWebIdentityResult": {
                "SubjectFromWebIdentityToken": "amzn1.account.AF6RHO7KZU5XRVQJGXK6HB56KR2A",
                "Audience": "client.5498841531868486423.1548@apps.example.com",
                "AssumedRoleUser": {
                    "Arn": role_arn,
                    "AssumedRoleId": metadata.key.id,
                },
                "Credentials": {
                    "SessionToken": metadata.key.token,
                    "SecretAccessKey": metadata.key.secret,
                    "Expiration": metadata.expires.timestamp(),  # In unix seconds
                    "AccessKeyId": metadata.key.id,
                },
                "SourceIdentity": "SourceIdentityValue",
                "Provider": "www.amazon.com",
            },
            "ResponseMetadata": {"RequestId": "ad4156e9-bce1-11e2-82e6-6b6efEXAMPLE"},
        }
    }
    return {"text": json.dumps(response), "status_code": 200}


def sts_failure_response(status: int, error_body: Optional[str]) -> Dict[str, Any]:
    return {"text": error_body, "status_code": status}


def aws_error(code: str) -> str:
    return json.dumps(
        {
            "Error": {
                "Code": code,
                "Message": f"Error message for {code}",
                "Type": "Something",
            },
            "RequestId": uuid.uuid4().hex,
        }
    )


@pytest.fixture
def web_identity_token() -> str:
    return f"Random_token_file_stuff_{uuid.uuid4().hex}"


@contextmanager
def mock_web_identity_token_file(
    filename: str, file_content: str
) -> Generator[None, None, None]:
    orig_open = open

    def open_patch(*args: Any, **kwargs: Any) -> Any:
        if args[0] == filename:
            return mock_open(read_data=file_content)(*args, **kwargs)
        return orig_open(*args, **kwargs)

    with patch("builtins.open", open_patch):
        yield


@pytest.fixture
def web_identity_token_filename(web_identity_token: str) -> Generator[str, None, None]:
    mock_filename = f"/tmp/token.{uuid.uuid4().hex}"
    with mock_web_identity_token_file(mock_filename, web_identity_token):
        yield mock_filename


@pytest.fixture
def role_arn() -> str:
    return f"arn:aws:iam::1234567890:role/{uuid.uuid4().hex}"


@pytest.fixture
def assume_role_web_identity_token_env_config(
    monkeypatch: MonkeyPatch,
    web_identity_token_filename: str,
    role_arn: str,
) -> None:
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", web_identity_token_filename)
    monkeypatch.setenv("AWS_ROLE_ARN", role_arn)


class WebIdentityServer:
    def __init__(self) -> None:
        self.port = 0
        self.status = 200
        self.text = ""
        self.requests: List[web.Request] = []

    async def handler(self, request: web.Request) -> web.Response:
        self.requests.append(request)
        return web.json_response(text=self.text, status=self.status)

    def set_response(self, *, text: str, status_code: int) -> None:
        self.text = text
        self.status = status_code


@pytest.fixture
async def web_identity_server() -> AsyncGenerator[WebIdentityServer, None]:
    server = WebIdentityServer()
    app = web.Application()
    app.add_routes([web.post("/", server.handler)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield server
    await runner.cleanup()


@pytest.fixture
def assume_role_web_identity_credentials(
    assume_role_web_identity_token_env_config: None,
    web_identity_server: WebIdentityServer,
) -> AssumeRoleWithWebIdentityEnvironmentCredentials:
    return AssumeRoleWithWebIdentityEnvironmentCredentials(
        sts_endpoint=URL("http://localhost").with_port(web_identity_server.port)
    )


@pytest.fixture
def fake_retry_sleep(
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
) -> List[float]:
    delays: List[float] = []
    orig_delays = assume_role_web_identity_credentials.retry_config.delays

    class FakeSleepRetries(RetryConfig):
        def delays(self) -> Iterable[Seconds]:
            yield from orig_delays()

        def _time_monotonic(self) -> float:
            return super()._time_monotonic() + sum(delays)

        async def _sleep(self, delay: float) -> None:
            delays.append(delay)
            await super()._sleep(0)

    assume_role_web_identity_credentials.retry_config = FakeSleepRetries()
    return delays


@pytest.mark.parametrize(
    "env_var_missing", ["AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN"]
)
def test_assume_role_web_identity_missing_config_is_disabled(
    monkeypatch: MonkeyPatch,
    assume_role_web_identity_token_env_config: None,
    env_var_missing: str,
) -> None:
    monkeypatch.delenv(env_var_missing, raising=False)
    assert AssumeRoleWithWebIdentityEnvironmentCredentials().is_disabled() is True


def test_assume_role_web_identity_sts_endpoint(
    monkeypatch: MonkeyPatch,
    assume_role_web_identity_token_env_config: None,
) -> None:
    sts_endpoint = AssumeRoleWithWebIdentityEnvironmentCredentials().sts_endpoint
    assert sts_endpoint is not None
    assert sts_endpoint.scheme == "https"
    assert sts_endpoint.host == "sts.amazonaws.com"

    monkeypatch.setenv("AWS_STS_REGIONAL_ENDPOINTS", "regional")
    for region in ["eu-central-1", "us-east-1"]:
        monkeypatch.setenv("AWS_DEFAULT_REGION", region)
        sts_endpoint = AssumeRoleWithWebIdentityEnvironmentCredentials().sts_endpoint
        assert sts_endpoint is not None
        assert sts_endpoint.scheme == "https"
        assert sts_endpoint.host == f"sts.{region}.amazonaws.com"


def test_assume_role_web_identity_role_session_name(
    monkeypatch: MonkeyPatch,
    assume_role_web_identity_token_env_config: None,
) -> None:
    url = AssumeRoleWithWebIdentityEnvironmentCredentials().url
    assert url is not None
    generated_role = url.query.get("RoleSessionName")
    assert generated_role is not None
    assert generated_role != ""

    role_name = uuid.uuid4().hex
    monkeypatch.setenv("AWS_ROLE_SESSION_NAME", role_name)

    url = AssumeRoleWithWebIdentityEnvironmentCredentials().url
    assert url is not None
    assert url.query.get("RoleSessionName") == role_name


def test_assume_role_web_identity_role_url_parameters(
    assume_role_web_identity_token_env_config: None,
    web_identity_token: str,
    role_arn: str,
) -> None:
    creds = AssumeRoleWithWebIdentityEnvironmentCredentials()
    assert creds.is_disabled() is False
    assert creds.url is not None

    used_session_name = creds.url.query.get("RoleSessionName")
    assert dict(creds.url.query) == {
        "Action": "AssumeRoleWithWebIdentity",
        "RoleArn": role_arn,
        "WebIdentityToken": web_identity_token,
        "RoleSessionName": used_session_name,
        "Version": "2011-06-15",
    }


def test_assume_role_web_identity_role_token_reread(
    assume_role_web_identity_token_env_config: None,
    web_identity_token: str,
    web_identity_token_filename: str,
) -> None:
    creds = AssumeRoleWithWebIdentityEnvironmentCredentials()
    assert creds.is_disabled() is False
    assert creds.url is not None
    assert creds.url.query.get("WebIdentityToken") == web_identity_token

    new_token = uuid.uuid4().hex
    with mock_web_identity_token_file(web_identity_token_filename, new_token):
        assert creds.url is not None
        assert creds.url.query.get("WebIdentityToken") == new_token


async def test_assume_role_web_identity_ok(
    http: HttpImplementation,
    web_identity_server: WebIdentityServer,
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    role_arn: str,
) -> None:
    metadata = Metadata(
        key=Key(
            id="id:" + role_arn,
            secret=uuid.uuid4().hex,
            token=uuid.uuid4().hex,
        ),
        expires=datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=1),
    )
    web_identity_server.set_response(**sts_mock_response(role_arn, metadata))

    assert await assume_role_web_identity_credentials.get_key(http) == metadata.key

    requests = web_identity_server.requests
    assert len(requests) == 1 and requests[0].headers["Accept"] == "application/json"


async def test_assume_role_web_identity_caching(
    http: HttpImplementation,
    web_identity_server: WebIdentityServer,
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    role_arn: str,
) -> None:
    metadata = Metadata(
        key=Key(
            id="id:" + role_arn,
            secret=uuid.uuid4().hex,
            token=uuid.uuid4().hex,
        ),
        expires=datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=1),
    )
    web_identity_server.set_response(**sts_mock_response(role_arn, metadata))

    assert await assume_role_web_identity_credentials.get_key(http) == metadata.key
    assert len(web_identity_server.requests) == 1

    assert await assume_role_web_identity_credentials.get_key(http) == metadata.key
    assert (
        len(web_identity_server.requests) == 1
    ), "Credentials should have been retrieved from cache on second time"

    assert (
        assume_role_web_identity_credentials._metadata is not None
    ), "Assumptions of internal state failed."
    assume_role_web_identity_credentials._metadata = Metadata(
        key=assume_role_web_identity_credentials._metadata.key,
        expires=datetime.datetime.now(datetime.timezone.utc),
    )
    assert await assume_role_web_identity_credentials.get_key(http) == metadata.key
    assert (
        len(web_identity_server.requests) == 2
    ), "Credentials should be re-retrieved from api after they expired"


@pytest.mark.parametrize(
    "error",
    [(400, "IDPCommunicationError"), (400, "ExpiredToken")],
)
async def test_assume_role_web_identity_credentials_sts_request_transient_api_errors(
    http: HttpImplementation,
    web_identity_server: WebIdentityServer,
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    error: Tuple[int, str],
    fake_retry_sleep: List[float],
) -> None:
    web_identity_server.set_response(
        **sts_failure_response(error[0], aws_error(error[1]))
    )
    with pytest.raises(TooManyRetries) as e:
        await assume_role_web_identity_credentials.get_key(http)
    assert isinstance(e.value.__cause__, AssumeRoleWebIdentityClientException)
    assert e.value.__cause__.status == error[0]
    assert e.value.__cause__.sts_code == error[1]
    assert len(web_identity_server.requests) > 1  # retries
    assert (
        sum(fake_retry_sleep[:-1])
        <= assume_role_web_identity_credentials.retry_config.time_limit_secs
    )


@pytest.mark.parametrize(
    "error",
    [
        (403, "IDPRejectedClaim"),
        (400, "InvalidIdentityToken"),
        (400, "MalformedPolicyDocument"),
        (400, "PackedPolicyTooLarge"),
        (403, "RegionDisabled"),
    ],
)
async def test_assume_role_web_identity_credentials_sts_request_non_transient_api_errors(
    http: HttpImplementation,
    web_identity_server: WebIdentityServer,
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    error: Tuple[int, str],
    fake_retry_sleep: List[float],
) -> None:
    web_identity_server.set_response(
        **sts_failure_response(error[0], aws_error(error[1]))
    )
    with pytest.raises(AssumeRoleWebIdentityClientException) as e:
        await assume_role_web_identity_credentials.get_key(http)
    assert e.value.status == error[0]
    assert e.value.sts_code == error[1]
    assert len(web_identity_server.requests) == 1  # no retries
    assert sum(fake_retry_sleep) == 0


@pytest.mark.parametrize("error_status_code", [400, 403, 429, 500, 503, 504])
@pytest.mark.parametrize(
    "error_body",
    [
        "non json body",
        "<no-aws-xml><stuff>here</stuff></no-aws-xml>",
        '{"foo": "bar"}',
        '["foo"]',
        None,
    ],
)
async def test_assume_role_web_identity_credentials_sts_request_non_api_errors(
    http: HttpImplementation,
    web_identity_server: WebIdentityServer,
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    error_status_code: int,
    error_body: Optional[str],
    fake_retry_sleep: List[float],
) -> None:
    web_identity_server.set_response(
        **sts_failure_response(error_status_code, error_body)
    )
    with pytest.raises(TooManyRetries) as e:
        await assume_role_web_identity_credentials.get_key(http)
    assert isinstance(e.value.__cause__, AssumeRoleWebIdentityHTTPStatusException)
    assert e.value.__cause__.status == error_status_code
    assert e.value.__cause__.body == (error_body or "").encode("utf8")
    assert len(web_identity_server.requests) > 1  # retries
    assert (
        sum(fake_retry_sleep[:-1])
        <= assume_role_web_identity_credentials.retry_config.time_limit_secs
    )


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientConnectionError("some connection error"),
        aiohttp.ServerConnectionError("some other connection error"),
        aiohttp.ServerTimeoutError("timeout error"),
    ],
)
async def test_assume_role_web_identity_credentials_sts_request_connection_error(
    assume_role_web_identity_credentials: AssumeRoleWithWebIdentityEnvironmentCredentials,
    error: Exception,
    fake_retry_sleep: List[float],
) -> None:
    class FailingSession(ClientSession):
        request_count = 0

        async def _request(self, *args: Any, **kwargs: Any) -> Any:
            self.request_count += 1
            raise error

    session = FailingSession()
    http = AIOHTTP(session)

    with pytest.raises(TooManyRetries) as e:
        await assume_role_web_identity_credentials.get_key(http)
    assert repr(e.value.__cause__) == repr(error)

    assert session.request_count > 1  # retries
    assert (
        sum(fake_retry_sleep[:-1])
        <= assume_role_web_identity_credentials.retry_config.time_limit_secs
    )
