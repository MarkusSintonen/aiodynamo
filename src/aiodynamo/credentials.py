from __future__ import annotations

import abc
import asyncio
import configparser
import datetime
import json
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import Callable, Optional, Sequence, TypeVar, cast

from yarl import URL

from .errors import BrokenThrottleConfig
from .http.types import HttpImplementation, Request, RequestFailed
from .models import ExponentialBackoffRetry, RetryConfig, RetryTimeout
from .types import Timeout
from .utils import logger, parse_amazon_timestamp


@dataclass(frozen=True)
class Key:
    id: str
    secret: str = field(repr=False)
    token: Optional[str] = field(repr=False, default=None)


class Credentials(metaclass=abc.ABCMeta):
    @classmethod
    def auto(cls) -> ChainCredentials:
        """
        Return the default credentials loader chain.
        """
        return ChainCredentials(
            candidates=[
                EnvironmentCredentials(),
                FileCredentials(),
                ContainerMetadataCredentials(),
                InstanceMetadataCredentials(),
            ]
        )

    @abc.abstractmethod
    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        """
        Return a Key if one could be found.
        """

    @abc.abstractmethod
    def invalidate(self) -> bool:
        """
        Invalidate the credentials if possible and return whether credentials
        got invalidated or not.
        """

    @abc.abstractmethod
    def is_disabled(self) -> bool:
        """
        Indicate if this credentials provider is disabled. Used by ChainCredentials
        to ignore providers that won't ever find a key.

        If the status could change over the lifetime of a program, return True.
        """


class Disabled(Exception):
    pass


@dataclass(frozen=True)
class StaticCredentials(Credentials):
    """
    Static credentials provided in Python.
    """

    key: Optional[Key]

    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        return self.key

    def invalidate(self) -> bool:
        return False

    def is_disabled(self) -> bool:
        return False


class ChainCredentials(Credentials):
    """
    Chains multiple credentials providers together, trying them
    in order. Returns the first key found. Exceptions are suppressed.
    """

    candidates: Sequence[Credentials]

    def __init__(self, candidates: Sequence[Credentials]) -> None:
        self.candidates = [
            candidate for candidate in candidates if not candidate.is_disabled()
        ]

    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        for candidate in self.candidates:
            try:
                key = await candidate.get_key(http)
            except:
                logger.exception("Candidate %r failed", candidate)
                continue
            if key is None:
                logger.debug("Candidate %r didn't find a key", candidate)
            else:
                logger.debug("Candidate %r found a key %r", candidate, key)
                return key
        return None

    def invalidate(self) -> bool:
        return any(candidate.invalidate() for candidate in self.candidates)

    def is_disabled(self) -> bool:
        return not self.candidates


class EnvironmentCredentials(Credentials):
    """
    Loads the credentials from the environment.
    """

    key: Optional[Key]

    def __init__(self) -> None:
        try:
            self.key = Key(
                os.environ["AWS_ACCESS_KEY_ID"],
                os.environ["AWS_SECRET_ACCESS_KEY"],
                os.environ.get("AWS_SESSION_TOKEN", None),
            )
        except KeyError:
            self.key = None

    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        return self.key

    def invalidate(self) -> bool:
        return False

    def is_disabled(self) -> bool:
        return self.key is None


# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
class FileCredentials(Credentials):
    """
    Loads the credentials from an AWS credentials file
    """

    key: Optional[Key]

    def __init__(
        self,
        *,
        path: Optional[Path] = None,
        profile_name: Optional[str] = None,
    ) -> None:
        if profile_name is None:
            profile_name = cast(str, os.environ.get("AWS_PROFILE", "default"))
        if path is None:
            path = Path.home().joinpath(".aws", "credentials")
        self.key = None
        if not path.is_file():
            return

        parser = configparser.RawConfigParser()
        try:
            parser.read(path)
        except configparser.Error:
            logger.exception("Found credentials file %r but parsing failed", path)
            return

        try:
            profile = parser[profile_name]
        except KeyError:
            logger.exception(
                "Profile %r not found in credentials file %r", profile_name, path
            )
            return

        try:
            self.key = Key(
                id=profile["aws_access_key_id"],
                secret=profile["aws_secret_access_key"],
                token=profile.get("aws_session_token", None),
            )
        except KeyError:
            logger.exception(
                "Profile %r in %r does not contain credentials", profile_name, path
            )

    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        return self.key

    def invalidate(self) -> bool:
        return False

    def is_disabled(self) -> bool:
        return self.key is None


class Refresh(Enum):
    required = auto()
    soon = auto()
    not_required = auto()


EXPIRES_SOON_THRESHOLD = datetime.timedelta(minutes=15)
EXPIRED_THRESHOLD = datetime.timedelta(minutes=10)


@dataclass(frozen=True)
class Metadata:
    key: Key
    expires: datetime.datetime

    def check_refresh(self) -> Refresh:
        now = datetime.datetime.now(datetime.timezone.utc)
        if now >= self.expires:
            return Refresh.required
        diff = self.expires - now
        if diff < EXPIRED_THRESHOLD:
            return Refresh.required
        if diff < EXPIRES_SOON_THRESHOLD:
            return Refresh.soon
        return Refresh.not_required


class MetadataCredentials(Credentials, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    async def fetch_metadata(self, http: HttpImplementation) -> Metadata:
        pass

    def __post_init__(self) -> None:
        self._metadata: Optional[Metadata] = None
        self._refresher: Optional[asyncio.Task[None]] = None

    async def get_key(self, http: HttpImplementation) -> Optional[Key]:
        if self.is_disabled():
            logger.debug("%r is disabled", self)
            return None
        refresh = self._check_refresh()
        logger.debug("refresh status %r", refresh)
        if refresh is Refresh.required:
            if self._refresher is None:
                logger.debug("starting mandatory refresh")
                self._refresher = asyncio.create_task(self._refresh(http))
            else:
                logger.debug("re-using refresh")
            try:
                await self._refresher
            finally:
                self._refresher = None
        elif refresh is Refresh.soon:
            if self._refresher is None:
                logger.debug("starting early refresh")
                self._refresher = asyncio.create_task(self._refresh(http))
            else:
                logger.debug("already refreshing")
        if self._metadata:
            return self._metadata.key
        return None

    def invalidate(self) -> bool:
        logger.debug("%r invalidated", self)
        self._metadata = None
        return True

    def _check_refresh(self) -> Refresh:
        if self._metadata is None:
            return Refresh.required
        return self._metadata.check_refresh()

    async def _refresh(self, http: HttpImplementation) -> None:
        self._metadata = await self.fetch_metadata(http)
        logger.debug("fetched metadata %r", self._metadata)


T = TypeVar("T")
U = TypeVar("U")


def and_then(thing: Optional[T], mapper: Callable[[T], U]) -> Optional[U]:
    if thing is not None:
        return mapper(thing)
    return thing


# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-metadata-endpoint.html
@dataclass
class ContainerMetadataCredentials(MetadataCredentials):
    """
    Loads credentials from the ECS container metadata endpoint.
    """

    timeout: Timeout = 2
    max_attempts: int = 3
    base_url: URL = URL("http://169.254.170.2")
    relative_uri: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", None
        )
    )
    full_uri: Optional[URL] = field(
        default_factory=lambda: and_then(
            os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI", None), URL
        )
    )
    auth_token: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "AWS_CONTAINER_AUTHORIZATION_TOKEN", None
        )
    )

    @property
    def url(self) -> Optional[URL]:
        if self.relative_uri is not None:
            return self.base_url.with_path(self.relative_uri)
        elif self.full_uri is not None:
            return self.full_uri
        else:
            return None

    def is_disabled(self) -> bool:
        return self.url is None

    async def fetch_metadata(self, http: HttpImplementation) -> Metadata:
        headers = and_then(self.auth_token, lambda token: {"Authorization": token})
        if self.url is None:
            raise Disabled()
        response = await fetch_with_retry_and_timeout(
            http=http,
            max_attempts=self.max_attempts,
            timeout=self.timeout,
            request=Request(
                method="GET", url=str(self.url), headers=headers, body=None
            ),
        )
        data = json.loads(response)
        return Metadata(
            key=Key(
                id=data["AccessKeyId"],
                secret=data["SecretAccessKey"],
                token=data.get("Token", None),
            ),
            expires=parse_amazon_timestamp(data["Expiration"]),
        )


# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instancedata-data-retrieval.html
@dataclass
class InstanceMetadataCredentials(MetadataCredentials):
    """
    Loads credentials from the EC2 instance metadata endpoint.
    """

    timeout: Timeout = 1
    max_attempts: int = 1
    base_url: URL = URL("http://169.254.169.254")
    disabled: bool = field(
        default_factory=lambda: os.environ.get(
            "AWS_EC2_METADATA_DISABLED", "false"
        ).lower()
        == "true"
    )

    def is_disabled(self) -> bool:
        return self.disabled

    async def fetch_metadata(self, http: HttpImplementation) -> Metadata:
        role_url = self.base_url.with_path(
            "/latest/meta-data/iam/security-credentials/"
        )
        role = (
            await fetch_with_retry_and_timeout(
                http=http,
                max_attempts=self.max_attempts,
                timeout=self.timeout,
                request=Request(
                    method="GET", url=str(role_url), headers=None, body=None
                ),
            )
        ).decode("utf-8")
        credentials_url = self.base_url.with_path(
            f"/latest/meta-data/iam/security-credentials/{role}"
        )
        raw_credentials = await fetch_with_retry_and_timeout(
            http=http,
            max_attempts=self.max_attempts,
            timeout=self.timeout,
            request=Request(
                method="GET", url=str(credentials_url), headers=None, body=None
            ),
        )
        credentials = json.loads(raw_credentials)
        return Metadata(
            key=Key(
                credentials["AccessKeyId"],
                credentials["SecretAccessKey"],
                credentials.get("Token", None),
            ),
            expires=parse_amazon_timestamp(credentials["Expiration"]),
        )


def sts_endpoint_default() -> URL:
    use_regional_endpoint = os.environ.get("AWS_STS_REGIONAL_ENDPOINTS") == "regional"
    if (
        use_regional_endpoint
        and (region := os.getenv("AWS_DEFAULT_REGION")) is not None
    ):
        return URL(f"https://sts.{region}.amazonaws.com")
    else:
        return URL("https://sts.amazonaws.com")


# https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
# https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
@dataclass
class AssumeRoleWithWebIdentityEnvironmentCredentials(MetadataCredentials):
    STS_SCHEMA_VERSION = "2011-06-15"
    # Retryable STS client errors
    # https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html#API_AssumeRoleWithWebIdentity_Errors
    STS_RETRYABLE_CLIENT_ERRORS = {"ExpiredToken", "IDPCommunicationError"}

    retry_config: RetryConfig = field(
        default_factory=lambda: ExponentialBackoffRetry(
            base_delay_secs=1.5, time_limit_secs=6
        )
    )

    web_identity_token_file: Optional[str] = field(
        default_factory=lambda: os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
    )
    role_arn: Optional[str] = field(default_factory=lambda: os.getenv("AWS_ROLE_ARN"))
    role_session_name: str = field(
        default_factory=lambda: os.getenv(
            "AWS_ROLE_SESSION_NAME", f"aiodynamo-session-{int(time())}"
        ),
    )
    sts_endpoint: URL = field(default_factory=sts_endpoint_default)

    def __post_init__(self) -> None:
        self.url_without_token = None

        if self.web_identity_token_file is None:
            return

        if self.role_arn is None:
            logger.error(
                "The the current environment is configured to assume role with web identity but has no "
                "role ARN configured. Ensure that the AWS_ROLE_ARN env var is set."
            )
            return

        self.url_without_token = self.sts_endpoint.with_query(
            {
                "Action": "AssumeRoleWithWebIdentity",
                "RoleArn": self.role_arn,
                "RoleSessionName": self.role_session_name,
                "Version": self.STS_SCHEMA_VERSION,
            }
        )
        super().__post_init__()

    @property
    def token(self) -> Optional[str]:
        if self.web_identity_token_file is None:
            return None
        else:
            try:
                with open(self.web_identity_token_file) as token_file:
                    return token_file.read()
            except IOError as e:
                raise AssumeRoleWebIdentityException(
                    "Failed to read web identity token file"
                ) from e

    @property
    def url(self) -> Optional[URL]:
        if self.url_without_token is not None and (token := self.token):
            return self.url_without_token.update_query({"WebIdentityToken": token})
        return None

    def is_disabled(self) -> bool:
        return self.url is None

    async def fetch_metadata(self, http: HttpImplementation) -> Metadata:
        if url := self.url:
            return await self.request_sts_credentials(url, http)
        raise Disabled()

    async def request_sts_credentials(
        self, url: URL, http: HttpImplementation
    ) -> Metadata:
        # Retry handling same as in aiodynamo Client.send_request
        exception: Optional[Exception] = None
        try:
            async for _ in self.retry_config.attempts():
                try:
                    request = Request(
                        method="POST",
                        url=str(url),
                        headers={"Accept": "application/json"},
                        body=None,
                    )
                    response = await http(request)
                    if 200 <= response.status <= 299:
                        return self.parse_credentials(response.body)

                    exception = AssumeRoleWebIdentityHTTPStatusException(
                        status=response.status, body=response.body
                    )
                    if response.status >= 500 or response.status == 429:
                        continue  # Retryable error codes
                    sts_api_exc = self.sts_api_exception(response.status, response.body)
                    if sts_api_exc is None:
                        continue  # Retry unknown api errors
                    exception = sts_api_exc
                    if sts_api_exc.sts_code in self.STS_RETRYABLE_CLIENT_ERRORS:
                        continue
                    raise sts_api_exc
                except asyncio.TimeoutError as exc:
                    exception = exc
                    continue
                except RequestFailed as exc:
                    exception = exc.inner
                    continue
        except RetryTimeout as retry_exc:
            cause = exception or retry_exc
            raise TooManyRetries("Too many retries to STS") from cause
        raise BrokenThrottleConfig()

    def sts_api_exception(
        self, status_code: int, raw_body: bytes
    ) -> Optional[AssumeRoleWebIdentityClientException]:
        try:
            error_json = json.loads(raw_body)
        except JSONDecodeError:
            return None

        if isinstance(error_json, dict) and (sts_error := error_json.get("Error")):
            return AssumeRoleWebIdentityClientException(
                sts_code=sts_error["Code"],
                sts_message=sts_error["Message"],
                status=status_code,
                body=raw_body,
            )
        return None

    def parse_credentials(self, raw_body: bytes) -> Metadata:
        resp = json.loads(raw_body)["AssumeRoleWithWebIdentityResponse"]
        credentials = resp["AssumeRoleWithWebIdentityResult"]["Credentials"]
        return Metadata(
            key=Key(
                id=credentials["AccessKeyId"],
                secret=credentials["SecretAccessKey"],
                token=credentials["SessionToken"],
            ),
            expires=datetime.datetime.fromtimestamp(
                float(credentials["Expiration"]),  # In unix seconds
                tz=datetime.timezone.utc,
            ),
        )


class TooManyRetries(Exception):
    pass


class AssumeRoleWebIdentityException(Exception):
    pass


class AssumeRoleWebIdentityHTTPStatusException(AssumeRoleWebIdentityException):
    def __init__(self, *, status: int, body: bytes, message: Optional[str] = None):
        super().__init__(
            message
            or f"STS AssumeRoleByWebIdentity request failed. status: {status}, body: {repr(body)}"
        )
        self.status = status
        self.body = body


class AssumeRoleWebIdentityClientException(AssumeRoleWebIdentityHTTPStatusException):
    def __init__(self, *, status: int, body: bytes, sts_code: str, sts_message: str):
        exc_msg = f"STS AssumeRoleByWebIdentity request failed. sts_code: {sts_code}, sts_message: {sts_message}"
        super().__init__(status=status, body=body, message=exc_msg)
        self.sts_code = sts_code
        self.sts_message = sts_code


async def fetch_with_retry_and_timeout(
    *,
    http: HttpImplementation,
    max_attempts: int,
    timeout: Timeout,
    request: Request,
) -> bytes:
    exception: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            response = await asyncio.wait_for(http(request), timeout)
        except asyncio.TimeoutError:
            logger.debug("timed out talking to metadata service")
            continue
        except RequestFailed as exc:
            logger.debug("request to metadata service failed")
            exception = exc.inner
            continue
        logger.debug(
            "fetched metadata %s (%s bytes)", response.status, len(response.body)
        )
        if response.status == 200:
            return response.body
    if exception is not None:
        raise exception
    raise TooManyRetries()
