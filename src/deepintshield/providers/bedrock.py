from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from ..errors import ErrorCode, _dependency_error
from ..transport import connection_headers

if TYPE_CHECKING:
    from ..client import DeepintShield


def build_client(shield: "DeepintShield", *, region_name: str | None = None, **kwargs: Any):
    """Return a boto3 ``bedrock-runtime`` client routed through the gateway."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise _dependency_error(
            "Install boto3: pip install 'deepintshield[bedrock]'",
            code=ErrorCode.PROVIDER_DEPENDENCY_MISSING,
            component="bedrock",
        ) from exc

    key = shield.api_key()
    client = boto3.client(
        service_name="bedrock-runtime",
        endpoint_url=kwargs.pop("endpoint_url", shield.bedrock_endpoint_url()),
        region_name=region_name or os.getenv("AWS_REGION", "us-west-2"),
        aws_access_key_id=kwargs.pop("aws_access_key_id", key),
        aws_secret_access_key=kwargs.pop("aws_secret_access_key", key),
        **kwargs,
    )
    headers = connection_headers(shield)

    def _inject_headers(request, **_kwargs):
        for name, value in headers.items():
            request.headers.add_header(name, value)

    client.meta.events.register_first("before-sign.bedrock-runtime.*", _inject_headers)
    return client
