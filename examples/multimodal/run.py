"""Native OpenAI SDK examples for the gateway's 29 provider identities.

Importing this module does not import SDKs, read environment variables, create
clients, or make requests. Provider operations below describe these examples,
not an exhaustive model capability catalog. Select a model that supports the
chosen input and operation; text examples are not multimodal certification.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from typing import Any
from urllib.parse import urlparse
import zlib


PROVIDERS = (
    "openai", "anthropic", "azure", "bedrock", "bedrock-mantle", "gemini",
    "vertex", "cohere", "mistral", "deepseek", "groq", "cerebras", "xai",
    "openrouter", "huggingface", "fireworks", "parasail", "nebius",
    "perplexity", "ollama", "vllm", "sgl", "replicate", "sarvam", "wafer",
    "opencode-go", "opencode-zen", "elevenlabs", "runway",
)
OPERATIONS = ("text", "vision", "pdf", "image", "speech", "transcription", "video", "file")
STREAM_OPERATIONS = ("text", "vision", "pdf")

# Dedicated operations mirror core/schemas/provider_registry.go. Vision/PDF are
# deliberately narrower: a Chat/Responses route alone is not evidence that an
# adapter/model understands images or documents. The deployed model must still
# advertise the selected modality. No model names are inferred or substituted.
_OPERATION_PROVIDERS = {
    "text": frozenset(PROVIDERS) - {"elevenlabs", "runway"},
    "vision": frozenset({"openai", "anthropic", "azure", "bedrock", "gemini",
                         "vertex", "cohere", "huggingface", "mistral", "nebius",
                         "sgl", "xai", "replicate", "wafer", "openrouter", "ollama",
                         "vllm", "fireworks", "parasail", "bedrock-mantle", "perplexity",
                         "opencode-go", "opencode-zen", "deepseek", "groq", "cerebras", "sarvam"}),
    "pdf": frozenset({"openai", "azure", "anthropic", "bedrock", "gemini", "vertex"}),
    "image": frozenset({"openai", "azure", "bedrock", "gemini", "vertex",
                        "huggingface", "nebius", "replicate", "runway", "xai"}),
    "speech": frozenset({"openai", "azure", "gemini", "groq", "huggingface", "sarvam", "elevenlabs"}),
    "transcription": frozenset({"openai", "azure", "gemini", "groq", "huggingface",
                                "mistral", "vllm", "sarvam", "elevenlabs"}),
    "video": frozenset({"openai", "azure", "gemini", "vertex", "replicate", "runway"}),
    # This example USES the uploaded file ID in Responses before deleting it.
    # Other Files adapters do not all translate file_id into inference input;
    # Bedrock additionally needs S3 configuration outside this example's API.
    "file": frozenset({"openai", "azure"}),
}
PROVIDER_OPERATIONS = {
    provider: tuple(operation for operation in OPERATIONS if provider in _OPERATION_PROVIDERS[operation])
    for provider in PROVIDERS
}

MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
VIDEO_POLL_TIMEOUT_SECONDS = 180.0
VIDEO_POLL_INTERVAL_SECONDS = 2.0
VIDEO_MAX_POLLS = 90

_PROMPTS = {
    "text": "Reply with one short sentence about a clear blue sky.",
    "vision": "What is the dominant color in this image? Reply with the color name.",
    "pdf": "What is the color of the square on the PDF page? Reply with the color name.",
    "image": "A simple blue square centered on a white background, no text.",
    "speech": "This is a short synthetic speech example.",
    "video": "A calm ocean under a clear blue sky, a short static camera shot.",
    "file": "What is the color of the square on the uploaded PDF page? Reply with the color name.",
}


def blue_png() -> bytes:
    """A real, decodable 128 by 128 RGB PNG, made entirely with the stdlib."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    pixels = (b"\x00" + b"\x00\x00\xff" * 128) * 128
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 128, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def blue_pdf() -> bytes:
    """One safe PDF page with an embedded blue RGB image, not a fake PDF blob."""
    pixels = zlib.compress(b"\x00\x00\xff" * 128 * 128)
    content = b"q 160 0 0 160 20 20 cm /Square Do Q\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /XObject << /Square 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
        b"<< /Type /XObject /Subtype /Image /Width 128 /Height 128 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length "
        + str(len(pixels)).encode() + b" >>\nstream\n" + pixels + b"\nendstream",
    ]
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start, count = len(data), len(objects) + 1
    data.extend(f"xref\n0 {count}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    return bytes(data)


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _selection(provider: str, operation: str, model: str, stream: bool) -> str:
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider; use --list for the 29 supported identities")
    if operation not in OPERATIONS or operation not in PROVIDER_OPERATIONS[provider]:
        raise ValueError(f"The {operation!r} example is not supported for {provider}; supported examples: "
                         + ", ".join(PROVIDER_OPERATIONS[provider]))
    if not isinstance(model, str) or not model.strip():
        raise ValueError("An explicit model is required; select one with the requested capability")
    model = model.strip()
    prefix = model.split("/", 1)[0]
    if "/" in model and prefix in PROVIDERS:
        if prefix != provider:
            raise ValueError("The model's provider prefix conflicts with --provider; qualify nested upstream IDs explicitly")
        if not model[len(provider) + 1:]:
            raise ValueError("The provider-qualified model must include a model ID")
        qualified = model
    else:
        qualified = f"{provider}/{model}"
    if stream and operation not in STREAM_OPERATIONS:
        raise ValueError("These examples support --stream only for text, vision and pdf")
    return qualified


def _input(operation: str, input_path: str | Path | None) -> tuple[str, bytes, str] | None:
    if input_path is None:
        if operation == "vision":
            return "blue.png", blue_png(), "image/png"
        if operation in {"pdf", "file"}:
            return "blue.pdf", blue_pdf(), "application/pdf"
        if operation == "transcription":
            raise ValueError("transcription requires --input containing real audio")
        return None
    if operation not in {"vision", "pdf", "transcription", "video", "file"}:
        raise ValueError(f"{operation} does not accept --input in this example")
    path = Path(input_path)
    with path.open("rb") as source:
        data = source.read(MAX_INPUT_BYTES + 1)
    if not data or len(data) > MAX_INPUT_BYTES:
        raise ValueError("Input must contain between 1 byte and 32 MiB")
    if operation in {"pdf", "file"}:
        if not data.startswith(b"%PDF-"):
            raise ValueError("pdf and file examples require a PDF input")
        return path.name, data, "application/pdf"
    if operation in {"vision", "video"}:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        elif operation == "vision" and data.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        elif operation == "vision" and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            raise ValueError("vision requires PNG/JPEG/GIF/WebP; video --input is a PNG/JPEG reference image")
        return path.name, data, mime
    audio_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
                   ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".mp4": "audio/mp4", ".webm": "audio/webm"}
    mime = audio_types.get(path.suffix.lower())
    if mime is None:
        raise ValueError("Use a real WAV, MP3, FLAC, OGG, M4A, MP4 or WebM audio fixture supported by the model")
    return path.name, data, mime


def _parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, dict) or any(not isinstance(key, str) for key in parameters):
        raise ValueError("parameters must be a JSON object of native operation kwargs")
    reserved = {"provider", "model", "messages", "input", "prompt", "file", "voice", "stream", "input_reference"}
    if reserved.intersection(parameters):
        raise ValueError("parameters cannot override the selected provider, model, input, prompt, voice or stream")
    for name in ("extra_body", "extra_query"):
        value = parameters.get(name)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"parameters.{name} must be a JSON object")
            if reserved.intersection(value):
                raise ValueError(f"parameters.{name} cannot override the selected provider, model or input")
    return dict(parameters)


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("The provider returned no nonempty text output")
    return value.strip()


def _assert_blue(text: str) -> None:
    """A refusal or a negated guess is not success on the known blue fixture."""
    normalized = text.casefold().replace("\u2019", "'")
    refusal = re.search(r"\b(?:cannot|can't|unable|couldn't|unsure|unknown|maybe|might|perhaps|probably|possibly)\b|"
                        r"\b(?:do not|don't|does not|doesn't) (?:have|see|know)\b|"
                        r"\bnot (?:able|sure|possible)\b", normalized)
    negated = re.search(r"\b(?:not|isn't|isnt|no)\s+(?:(?:a|the|primarily|mostly|really)\s+)*blue\b", normalized)
    ambiguous = re.search(r"\b(?:blue\s+or\b|or\s+blue\b)", normalized)
    other_colors = r"(?:red|green|yellow|orange|purple|pink|brown|cyan|magenta)"
    mixed = re.search(r"\b(?:blue\s+(?:and|/)\s+(?:also\s+)?" + other_colors
                      + r"|" + other_colors + r"\s+(?:and|/)\s+(?:also\s+)?blue)\b", normalized)
    if not re.search(r"\bblue\b", normalized) or refusal or negated or ambiguous or mixed:
        raise RuntimeError("The synthetic blue fixture did not produce a clear, non-refusal blue answer")


def _response_text(response: Any) -> str:
    if _get(response, "error") or _get(response, "status") != "completed":
        raise RuntimeError("The Responses request did not complete successfully")
    blocks = [block for item in (_get(response, "output") or []) for block in (_get(item, "content") or [])]
    if any(_get(block, "type") == "refusal" for block in blocks):
        raise RuntimeError("The Responses request returned a refusal instead of the requested answer")
    # Read the native structured output directly. Its output_text convenience
    # property can raise TypeError when a translated snapshot has output=None.
    text = "".join(_get(block, "text", "") for block in blocks if _get(block, "type") == "output_text")
    # The native Responses stream helper also uses the terminal snapshot as
    # its final value. Deltas cannot make a missing/empty final output valid.
    return _text(text)


def _chat(client: Any, model: str, content: Any, stream: bool, parameters: dict[str, Any]) -> str:
    result = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], stream=stream, **parameters)
    if not stream:
        choices = _get(result, "choices") or []
        if _get(result, "error") or not choices or _get(choices[0], "finish_reason") != "stop":
            raise RuntimeError("The Chat request did not finish with a complete text answer")
        message = _get(choices[0], "message")
        if _get(message, "refusal"):
            raise RuntimeError("The Chat request returned a refusal instead of the requested answer")
        return _text(_get(message, "content"))
    parts, completed = [], False
    try:
        for chunk in result:
            if _get(chunk, "error"):
                raise RuntimeError("The Chat stream reported an error")
            for choice in (_get(chunk, "choices") or []):
                delta = _get(choice, "delta")
                if _get(delta, "refusal"):
                    raise RuntimeError("The Chat stream returned a refusal instead of the requested answer")
                text = _get(delta, "content")
                if text:
                    parts.append(text)
                reason = _get(choice, "finish_reason")
                if reason is not None:
                    if reason != "stop":
                        raise RuntimeError("The Chat stream ended without a complete text answer")
                    completed = True
    finally:
        result.close()
    if not completed:
        raise RuntimeError("The Chat stream ended before its completion marker")
    return _text("".join(parts))


def _responses(client: Any, model: str, content: list[dict[str, Any]], stream: bool,
               parameters: dict[str, Any]) -> str:
    result = client.responses.create(model=model, input=[{"role": "user", "content": content}], stream=stream, **parameters)
    if not stream:
        return _response_text(result)
    completed_text = None
    try:
        for event in result:
            kind = _get(event, "type", "")
            if kind in {"error", "response.failed", "response.incomplete"} or _get(event, "error"):
                raise RuntimeError("The Responses stream reported an error or incomplete output")
            if kind in {"response.refusal.delta", "response.refusal.done"}:
                raise RuntimeError("The Responses stream returned a refusal instead of the requested answer")
            if kind == "response.output_text.delta":
                delta = _get(event, "delta")
                if not isinstance(delta, str):
                    raise RuntimeError("The Responses stream returned an invalid text delta")
            if kind == "response.completed":
                completed_text = _response_text(_get(event, "response"))
    finally:
        result.close()
    if completed_text is None:
        raise RuntimeError("The Responses stream ended before response.completed")
    return completed_text


def _binary(response: Any, kind: str) -> bytes:
    try:
        headers = _get(response, "headers") or _get(_get(response, "response"), "headers", {})
        content_type = headers.get("content-type", "").lower().split(";", 1)[0]
        if content_type and not (content_type.startswith(kind + "/") or content_type == "application/octet-stream"):
            raise RuntimeError("The provider returned an unexpected content type instead of binary media")
        data = bytearray()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            if len(data) + len(chunk) > MAX_OUTPUT_BYTES:
                raise RuntimeError("The provider returned oversized binary output")
            data.extend(chunk)
    finally:
        response.close()
    if not data:
        raise RuntimeError("The provider returned empty binary output")
    return bytes(data)


def _is_image(data: bytes) -> bool:
    # Validate an image header, without adding an image-decoding dependency or
    # pretending this signature check is a guardrail/OCR inspection.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return len(data) >= 33 and data[12:16] == b"IHDR" and all(struct.unpack(">II", data[16:24]))
    if data.startswith(b"\xff\xd8\xff"):
        return len(data) >= 4 and data.endswith(b"\xff\xd9")
    if data.startswith((b"GIF87a", b"GIF89a")):
        return len(data) >= 13
    return len(data) >= 20 and data.startswith(b"RIFF") and data[8:12] == b"WEBP"


def _artifact(data: bytes, output_path: str | Path | None) -> dict[str, Any]:
    result = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if output_path is not None:
        path = Path(output_path)
        # Do not overwrite existing inputs or user artifacts in an example.
        with path.open("xb") as target:
            target.write(data)
        result["output_path"] = str(path)
    return result


def _images(response: Any, output_path: str | Path | None) -> dict[str, Any]:
    images = _get(response, "data") or []
    if _get(response, "error") or not images:
        raise RuntimeError("Image generation returned no images")
    if output_path is not None and len(images) != 1:
        raise RuntimeError("--output requires one returned image; omit it for multiple artifact summaries")
    artifacts = []
    for image in images:
        encoded = _get(image, "b64_json")
        if encoded:
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Image generation returned invalid base64") from exc
            if not data or len(data) > MAX_OUTPUT_BYTES:
                raise RuntimeError("Image generation returned empty or oversized binary output")
            if not _is_image(data):
                raise RuntimeError("Image generation returned bytes without a supported image header")
            artifacts.append(_artifact(data, output_path))
        else:
            url = _get(image, "url")
            if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
                raise RuntimeError("Image generation returned neither image bytes nor an HTTP(S) artifact URL")
            if output_path is not None:
                raise RuntimeError("This model returned an artifact URL; omit --output to retain the URL without an external download")
            artifacts.append({"url": url})
    return {"images": len(images), **(artifacts[0] if len(artifacts) == 1 else {"artifacts": artifacts})}


def _video(client: Any, model: str, prompt: str, source: tuple[str, bytes, str] | None,
           output_path: str | Path | None, parameters: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "prompt": prompt, **parameters}
    if source is not None:
        kwargs["input_reference"] = source
    job = client.videos.create(**kwargs)
    video_id = _get(job, "id")
    if not isinstance(video_id, str) or not video_id:
        raise RuntimeError("Video creation returned no task ID")
    # Gateway IDs already carry their provider suffix. The native SDK quotes
    # path arguments, so preserve the entire returned ID, including slashes.
    deadline = time.monotonic() + VIDEO_POLL_TIMEOUT_SECONDS
    for _ in range(VIDEO_MAX_POLLS + 1):
        status = _get(job, "status")
        if _get(job, "error") or status not in {"queued", "in_progress", "completed"}:
            raise RuntimeError("Video generation failed or returned an unknown task status")
        if status == "completed":
            with client.videos.with_streaming_response.download_content(video_id) as response:
                data = _binary(response, "video")
            return {"video_id": video_id, **_artifact(data, output_path)}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(VIDEO_POLL_INTERVAL_SECONDS, remaining))
        job = client.videos.retrieve(video_id, timeout=min(30.0, max(0.1, deadline - time.monotonic())))
        if _get(job, "id") != video_id:
            raise RuntimeError("Video polling returned a different task ID")
    raise RuntimeError("Video generation did not complete within the bounded polling window; the remote task may continue")


def _file(client: Any, provider: str, model: str, prompt: str,
          source: tuple[str, bytes, str], parameters: dict[str, Any]) -> dict[str, Any]:
    # Azure's Responses upload guide uses assistants; user_data is not accepted
    # by that Files API. Keep OpenAI's native Responses purpose for OpenAI.
    uploaded = client.files.create(file=source, purpose="assistants" if provider == "azure" else "user_data",
                                   extra_body={"provider": provider})
    file_id = _get(uploaded, "id")
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError("File upload returned no ID; no other file will be deleted")
    primary_error = None
    try:
        text = _responses(client, model, [{"type": "input_text", "text": prompt},
                                         {"type": "input_file", "file_id": file_id}], False, parameters)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        # Clean up only the ID created by this call, even if inference fails.
        try:
            deleted = client.files.delete(file_id, extra_query={"provider": provider})
            if _get(deleted, "deleted") is not True or _get(deleted, "id") != file_id:
                raise RuntimeError("Uploaded example file deletion was not confirmed")
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            # add_note is available on Python 3.11+; the SDK also supports 3.10.
            note = getattr(primary_error, "add_note", None)
            if callable(note):
                note("Cleanup of the example's uploaded file failed: " + type(cleanup_error).__name__)
            primary_error.example_file_cleanup_failed = True
    return {"text": text, "file_id": file_id, "deleted": True}


def run_example(provider: str, operation: str, model: str, *, input_path: str | Path | None = None,
                output_path: str | Path | None = None, stream: bool = False, voice: str | None = None,
                prompt: str | None = None, client: Any = None,
                parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one explicitly selected example; injected native clients stay open.

    ValueError denotes unsupported selections/missing fixtures, RuntimeError an
    invalid or incomplete result. Native SDK/provider errors remain visible.
    Files uploads and file references do not imply inspection of file contents.
    """
    qualified = _selection(provider, operation, model, stream)
    options = _parameters(parameters)
    if operation == "speech" and (not isinstance(voice, str) or not voice.strip()):
        raise ValueError("speech requires an explicit --voice supported by the selected provider/model")
    if provider == "sarvam" and operation == "speech":
        language = (options.get("extra_body") or {}).get("language_code")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Sarvam speech requires parameters.extra_body.language_code, for example en-IN")
    if voice is not None and operation != "speech":
        raise ValueError("--voice applies only to the speech example")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("--prompt must contain nonempty text")
    source = _input(operation, input_path)
    if output_path is not None and Path(output_path).exists():
        raise ValueError("--output must be a new path; this example does not overwrite files")
    if output_path is not None and operation == "image" and options.get("n", 1) != 1:
        raise ValueError("--output supports one image; omit it to request multiple image artifact summaries")
    resources = ExitStack()
    result: dict[str, Any] = {"provider": provider, "operation": operation, "model": qualified,
                              "status": "completed", "stream": stream}
    instruction = prompt if prompt is not None else _PROMPTS.get(operation)
    try:
        if client is None:
            from deepintshield import DeepintShield
            from openai import OpenAI
            shield = resources.enter_context(DeepintShield.from_env())
            client = resources.enter_context(OpenAI(**shield.openai_config()))
        if operation in {"text", "vision"}:
            content: Any = instruction
            if operation == "vision":
                assert source is not None
                content = [{"type": "text", "text": instruction}, {"type": "image_url", "image_url": {
                    "url": f"data:{source[2]};base64," + base64.b64encode(source[1]).decode()}}]
            result["text"] = _chat(client, qualified, content, stream, options)
        elif operation == "pdf":
            assert source is not None
            result["text"] = _responses(client, qualified, [{"type": "input_text", "text": instruction}, {
                "type": "input_file", "filename": source[0],
                "file_data": "data:application/pdf;base64," + base64.b64encode(source[1]).decode()}], stream, options)
        elif operation == "image":
            result.update(_images(client.images.generate(model=qualified, prompt=instruction,
                                                        **{"n": 1, **options}), output_path))
        elif operation == "speech":
            with client.audio.speech.with_streaming_response.create(
                    model=qualified, input=instruction, voice=voice, **options) as response:
                result.update(_artifact(_binary(response, "audio"), output_path))
        elif operation == "transcription":
            assert source is not None
            kwargs = {"model": qualified, "file": source, **options}
            if prompt is not None:
                kwargs["prompt"] = prompt
            response = client.audio.transcriptions.create(**kwargs)
            if _get(response, "error"):
                raise RuntimeError("Transcription returned an error")
            result["text"] = _text(response if isinstance(response, str) else _get(response, "text"))
        elif operation == "video":
            result.update(_video(client, qualified, instruction, source, output_path, options))
        elif operation == "file":
            assert source is not None
            result.update(_file(client, provider, qualified, instruction, source, options))
        if operation in {"vision", "pdf", "file"} and input_path is None and prompt is None:
            _assert_blue(result["text"])
        if "text" in result and output_path is not None:
            result.update(_artifact(result["text"].encode("utf-8"), output_path))
        return result
    finally:
        resources.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--model", help="Explicit model or provider/model; use a model with the selected capability")
    parser.add_argument("--input", dest="input_path", help="Local image, PDF, audio, or video reference image")
    parser.add_argument("--output", dest="output_path", help="New output file; existing files are never overwritten")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--voice")
    parser.add_argument("--prompt")
    parser.add_argument("--parameters", help="JSON object of native operation kwargs; vendor options go in extra_body")
    parser.add_argument("--dry-run", action="store_true", help="Describe a selection without loading SDKs or making requests")
    parser.add_argument("--list", action="store_true", help="List provider/example combinations without making requests")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(PROVIDER_OPERATIONS, indent=2))
        return 0
    if not args.provider or not args.operation:
        parser.error("--provider and --operation are required unless --list is used")
    try:
        if args.parameters is not None:
            try:
                args.parameters = json.loads(args.parameters)
            except json.JSONDecodeError as exc:
                raise ValueError("--parameters must contain a JSON object") from exc
            if not isinstance(args.parameters, dict):
                raise ValueError("--parameters must contain a JSON object")
        _parameters(args.parameters)
        if args.dry_run:
            qualified = _selection(args.provider, args.operation, args.model or "<explicit-model-required>", args.stream)
            print(json.dumps({"provider": args.provider, "operation": args.operation, "model": qualified,
                              "stream": args.stream, "dry_run": True, "requires_input": args.operation == "transcription",
                              "requires_voice": args.operation == "speech", "model_capability_required": True}, indent=2))
        else:
            values = vars(args).copy()
            values.pop("dry_run")
            values.pop("list")
            print(json.dumps(run_example(**values), indent=2))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
