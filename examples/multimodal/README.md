<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 DeepIntShield contributors -->

# Native OpenAI multimodal examples

[run.py](run.py) uses the native OpenAI Python SDK through the gateway. Its
curated examples cover all **29 provider identities** using operations each
gateway adapter implements. A text example establishes text coverage; it does
not certify that provider's models for vision, documents or generated media.

The client needs the existing virtual key and gateway URL. Provider credentials,
model access and guardrail policy remain on the server. The runner does not
discover models, substitute a model, or add universal sampling/reasoning values.

## Setup and first run

Run these commands from the `deepintshield/` SDK directory. Install this checkout
with `pip install -e .`, or install a matching SDK release. The base package
includes the native OpenAI SDK; provider-specific extras are unnecessary for
these examples.

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="<virtual-key>"
export DEEPINTSHIELD_BASE_URL="http://localhost:8080"

# No credentials, SDK imports, requests or file writes in these two modes.
python examples/multimodal/run.py --list
python examples/multimodal/run.py \
  --provider openai --operation vision --model gpt-4o-mini --dry-run

# Sends an in-memory synthetic blue PNG to an enabled vision model.
python examples/multimodal/run.py \
  --provider openai --operation vision --model gpt-4o-mini

# Sends a synthetic one-page PDF through native Responses and consumes SSE.
python examples/multimodal/run.py \
  --provider anthropic --operation pdf --model claude-sonnet-4-5 --stream
```

Replace the example model IDs with models enabled for your account and virtual
key. Azure uses your deployment name; Bedrock may use an inference profile ID;
Vertex needs the configured project and location. Self-hosted providers such as
Ollama, vLLM and SGL need a reachable serving endpoint with the chosen model.
The virtual key must authorize the selected provider, model and operation.

`--model` is required for a real run. Both an upstream ID and a matching
`provider/model` ID are accepted. Nested upstream IDs are preserved: for
example, use `--provider replicate --model replicate/owner/model`. If an
upstream ID itself begins with another registered provider name, explicitly
prefix the selected gateway provider, such as `replicate/openai/<upstream-id>`.

These are direct inference examples. An agentic application additionally needs
the existing registered agent identity and its permissions; this runner does
not enroll or activate an agent.

Applications that already construct an OpenAI client can keep it. Set its usual
`OPENAI_BASE_URL=http://localhost:8080/v1` and `OPENAI_API_KEY` to the virtual key,
then pass that client into the checkout's example function:

```python
# Run from the SDK checkout directory; examples is not a published SDK API.
from openai import OpenAI
from examples.multimodal.run import run_example

with OpenAI() as client:
    result = run_example("anthropic", "pdf", "claude-sonnet-4-5", client=client)
    print(result["text"])
```

The example leaves a caller-supplied client open; the caller owns its lifecycle.

## Operations and inputs

| `--operation` | Native OpenAI method | Input and result |
| --- | --- | --- |
| `text` | `chat.completions.create` | Prompt text; nonempty completed text answer |
| `vision` | `chat.completions.create` | Inline `image_url` data URI plus prompt; text answer |
| `pdf` | `responses.create` | Inline `input_file.file_data` plus prompt; text answer |
| `image` | `images.generate` | Prompt; generated image bytes or an upstream artifact URL |
| `speech` | `audio.speech.create` | Text and explicit `--voice`; audio bytes |
| `transcription` | `audio.transcriptions.create` | Required local audio file as multipart input; transcript text |
| `video` | `videos.create`, `retrieve`, `download_content` | Prompt and optional PNG/JPEG reference image; completed video bytes |
| `file` | `files.create`, `responses.create`, `files.delete` | Upload a PDF, use its returned file ID, then delete that same uploaded file |

`text` and `vision` require a model supporting Chat Completions; `pdf` and `file`
require Responses. A model that only supports another API is not suitable for
that example. `--stream` is available for `text`, `vision` and `pdf`; the runner
consumes the full stream and prints the final result JSON. It treats incomplete,
empty or failed outputs as errors. Other operations use their ordinary native
request methods.

Without `--input`, vision uses a valid 128×128 blue PNG and PDF/file examples use
a valid single-page PDF containing a blue raster square. These fixtures are
generated in memory with Python's standard library and contain no personal
document content. The default fixture and prompt check whether the model
actually identifies blue. This checks model input handling, not OCR or PII
detection by guardrails.

To use your own input, supply `--input /path/to/file` and an appropriate
`--prompt`. Vision accepts PNG, JPEG, GIF or WebP; PDF/file inputs must be PDFs.
Transcription requires a real audio file supported by the selected model
(WAV, MP3, FLAC, OGG, M4A, MP4 or WebM). Video `--input` is a PNG/JPEG reference
image, not a video to analyze. The example limits local input to 32 MiB;
provider/model limits may be lower.

```bash
# Text input does not require a vision model.
python examples/multimodal/run.py \
  --provider deepseek --operation text --model '<enabled-chat-model>'

# Uses an actual voice available to your ElevenLabs account.
python examples/multimodal/run.py \
  --provider elevenlabs --operation speech --model eleven_multilingual_v2 \
  --voice '<voice-id>' --output generated-speech.mp3

python examples/multimodal/run.py \
  --provider elevenlabs --operation transcription --model scribe_v2 \
  --input /path/to/speech.wav

# Sarvam speech requires a language as well as a supported speaker.
python examples/multimodal/run.py \
  --provider sarvam --operation speech --model 'bulbul:v3' --voice '<speaker>' \
  --parameters '{"extra_body":{"language_code":"en-IN"}}'

# The gateway supplies the documented default image ratio for this model.
python examples/multimodal/run.py \
  --provider runway --operation image --model gen4_image

# Select a text-to-video model; provide --input if your model requires a reference.
python examples/multimodal/run.py \
  --provider runway --operation video --model gen4.5 --output generated-video.mp4

# The default PDF is uploaded, referenced in inference and deleted afterward.
python examples/multimodal/run.py \
  --provider openai --operation file --model gpt-4o-mini
```

`--output` writes to a new path and never overwrites an existing file. Without
it, binary results report their byte count and SHA-256 digest. Text results
include the answer and can optionally be saved as UTF-8. Binary output is limited
to 128 MiB. An image model may return a URL instead of bytes: the runner reports
that URL without fetching it, and `--output` then reports an explicit error.
Choose an output extension matching the format your model returns.

Video examples poll for up to 180 seconds with a bounded number of polls, then
download completed content through the gateway using the returned provider-scoped
ID. A timeout does not cancel the upstream task. File examples attempt cleanup
in `finally`, including after inference errors, and report unconfirmed deletion;
they never delete an unrelated existing file. Provider authentication, quota,
policy and model errors remain failures rather than successful example results.

`test_sdk.py` allows 240 seconds for video cases and 120 seconds for other live
cases; `--timeout` overrides this. Forced process termination cannot run upload
cleanup, so check the provider's file list if a file example times out.

The file example uses upload purpose `user_data` on OpenAI and `assistants` on
Azure, following the documented
[Azure PDF upload requirement](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/responses).
The provider must support the upload, Responses file input and deletion using
the configured account and API version.

Use `--parameters` with a JSON object to pass additional native method arguments
required by your selected model. For example, Sarvam speech needs
`{"extra_body":{"language_code":"en-IN"}}`, matching its current
[REST speech API](https://docs.sarvam.ai/api-reference/text-to-speech/convert).
A video model can receive caller-selected `size` and
`seconds` through the same option. Core example fields such as model, prompt,
input and stream remain controlled by their dedicated arguments.

For Runway, `gen4.5` supplies a gateway default ratio and duration; `gen4_turbo`
requires a reference image via `--input`. `gen4_image` supports text-to-image,
whereas `gen4_image_turbo` requires reference images supplied in
`extra_body.input_images` or the provider's native `referenceImages` shape.
Unknown model IDs do not receive guessed defaults. Select the model's actual
parameters from the gateway metadata or its provider documentation.

## All 29 provider identities

This table matches `PROVIDER_OPERATIONS` in the runner. Every listed operation
still requires a compatible model and configured credentials. Omission means
the runner does not provide that example; it is not an assertion that the
upstream provider has no such capability. In particular, Chat support alone
does not imply image or PDF support.

| Exact provider ID | Available examples |
| --- | --- |
| `openai` | text, vision, pdf, image, speech, transcription, video, file |
| `anthropic` | text, vision, pdf |
| `azure` | text, vision, pdf, image, speech, transcription, video, file |
| `bedrock` | text, vision, pdf, image |
| `bedrock-mantle` | text, vision |
| `gemini` | text, vision, pdf, image, speech, transcription, video |
| `vertex` | text, vision, pdf, image, video |
| `cohere` | text, vision |
| `mistral` | text, vision, transcription |
| `deepseek` | text, vision |
| `groq` | text, vision, speech, transcription |
| `cerebras` | text, vision |
| `xai` | text, vision, image |
| `openrouter` | text, vision |
| `huggingface` | text, vision, image, speech, transcription |
| `fireworks` | text, vision |
| `parasail` | text, vision |
| `nebius` | text, vision, image |
| `perplexity` | text, vision |
| `ollama` | text, vision |
| `vllm` | text, vision, transcription |
| `sgl` | text, vision |
| `replicate` | text, vision, image, video |
| `sarvam` | text, vision, speech, transcription |
| `wafer` | text, vision |
| `opencode-go` | text, vision |
| `opencode-zen` | text, vision |
| `elevenlabs` | speech, transcription |
| `runway` | image, video |

Vision availability is model-specific, including on the 27 providers with a
Chat route. A text-only model on any of those providers remains unsuitable for
the vision example. For example, Sarvam's `gemma4` vision access requires the
provider's per-key beta entitlement, and Cerebras vision availability depends
on its shared or dedicated model tier.

The dedicated-operation mapping follows the server's
[provider registry](../../../deepintshield_server/core/schemas/provider_registry.go).
Vision and PDF selections additionally follow the adapters' content conversions;
the exact deployed model must support the input. Wafer's
[Files API documentation](https://docs.wafer.ai/serverless/files-api) confirms
inline image data URIs for Chat. Its separate document-upload capability does
not establish this runner's Responses PDF or file-cleanup contract.

The shared OpenAI Chat handler also preserves inline images for the additional
vision paths documented by [OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding),
[Ollama](https://docs.ollama.com/api/openai-compatibility),
[vLLM](https://docs.vllm.ai/en/latest/features/multimodal_inputs/),
[Fireworks](https://docs.fireworks.ai/guides/querying-vision-language-models),
[Parasail](https://docs.parasail.io/parasail-docs/cookbooks/multi-modal),
[Bedrock Mantle](https://aws.amazon.com/blogs/machine-learning/introducing-grok-on-amazon-bedrock/),
[Groq](https://console.groq.com/docs/vision),
[DeepSeek](https://api-docs.deepseek.com/guides/vision/),
[Cerebras](https://inference-docs.cerebras.ai/capabilities/image-inputs), and
[Sarvam](https://docs.sarvam.ai/api/getting-started/models/open-source/gemma-4-31b).
The custom Perplexity conversion preserves the same image content documented
in its [media guide](https://docs.perplexity.ai/docs/sonar/media).
[OpenCode Go](https://opencode.ai/docs/go/) and
[OpenCode Zen](https://opencode.ai/docs/zen/) list vision-capable models on their
Chat endpoints; choose a model served on that endpoint and meet the provider's
account, client-identification and session requirements. These source checks
establish a compatible request path, not successful live inference on your key.

Other gateway operations, including embeddings, rerank, OCR, image editing,
batch jobs, realtime and provider-native extensions, are outside this runner.
See the [gateway integration guide](../../../deepintshield_server/docs/integrations/openai-sdk/all-providers.mdx)
for the broader operation contract.

## Run through `test_sdk.py`

Run the shared test runner from the **workspace root**, one directory above
`deepintshield/`. It imports the same examples and registers
`multimodal/<provider>/<operation>` cases.

The `DEEPINTSHIELD_TEST_MM_*` variables below are optional **test fixtures**,
not application configuration. Applications still connect with the virtual key
and gateway URL and pass their model, input and native request parameters as
ordinary SDK arguments. The standalone runner accepts those values as CLI
arguments; it does not require 29 provider environment configurations.

```bash
export DEEPINTSHIELD_VIRTUAL_KEY="<virtual-key>"
export DEEPINTSHIELD_BASE_URL="http://localhost:8080"
export DEEPINTSHIELD_TEST_MM_OPENAI_VISION_MODEL="gpt-4o-mini"
export DEEPINTSHIELD_TEST_MM_ANTHROPIC_PDF_MODEL="claude-sonnet-4-5"

python test_sdk.py --multimodal
python test_sdk.py --case multimodal/openai/vision --stream
```

`--multimodal` selects these cases; `--all` also includes them. Each case requires
an explicit `DEEPINTSHIELD_TEST_MM_<PROVIDER_NORMALIZED>_<OPERATION>_MODEL`.
Normalization uppercases the exact provider ID and replaces hyphens with
underscores: for example,
`DEEPINTSHIELD_TEST_MM_BEDROCK_MANTLE_TEXT_MODEL`.

Use the same prefix with `_INPUT`, `_OUTPUT`, `_VOICE`, `_PROMPT` or `_PARAMETERS`
for the corresponding runner arguments. `_PARAMETERS` contains a JSON object
of native request arguments. Transcription needs `_INPUT`; speech needs
`_VOICE`, and Sarvam speech also needs `extra_body.language_code`. For example:

```bash
export DEEPINTSHIELD_TEST_MM_ELEVENLABS_TRANSCRIPTION_MODEL="scribe_v2"
export DEEPINTSHIELD_TEST_MM_ELEVENLABS_TRANSCRIPTION_INPUT="/path/to/speech.wav"
export DEEPINTSHIELD_TEST_MM_ELEVENLABS_SPEECH_MODEL="eleven_multilingual_v2"
export DEEPINTSHIELD_TEST_MM_ELEVENLABS_SPEECH_VOICE="<voice-id>"
export DEEPINTSHIELD_TEST_MM_SARVAM_SPEECH_MODEL="bulbul:v3"
export DEEPINTSHIELD_TEST_MM_SARVAM_SPEECH_VOICE="<speaker>"
export DEEPINTSHIELD_TEST_MM_SARVAM_SPEECH_PARAMETERS='{"extra_body":{"language_code":"en-IN"}}'
```

`DEEPINTSHIELD_TEST_VK_<PROVIDER_NORMALIZED>` overrides the shared virtual key
for that provider. Unconfigured cases are explicit skips. A skipped case is
not proof of a working model, and offline native-transport tests do not certify
live model access. Streaming selection applies only to text, vision and PDF
cases; dedicated media cases keep their normal execution path.

For an explicit negative policy test, add `_EXPECT_GUARDRAIL_BLOCK=1` to the
same test-case prefix and supply a synthetic fixture that matches an
already active enforcing policy. The included [PDF fixture](fixtures/guardrail-visible-pii.pdf)
contains a fake SSN and a standard test card number in visible text. The included
[PNG fixture](fixtures/guardrail-image-metadata.png) contains the same synthetic
values in textual metadata. Neither contains a personal document:

```bash
export DEEPINTSHIELD_TEST_MM_OPENAI_PDF_MODEL="gpt-4o-mini"
export DEEPINTSHIELD_TEST_MM_OPENAI_PDF_INPUT="deepintshield/examples/multimodal/fixtures/guardrail-visible-pii.pdf"
export DEEPINTSHIELD_TEST_MM_OPENAI_PDF_EXPECT_GUARDRAIL_BLOCK=1
python test_sdk.py --case multimodal/openai/pdf
```

Only a native HTTP 403 carrying the structured `guardrail_blocked` code passes
that negative case. Successful inference, a model-access denial, quota error
or service failure does not pass. The report identifies the expected guardrail
rejection, and the test changes no policy. The ordinary blue fixtures contain
no PII and are intended for positive input-handling tests.

## Current guardrail boundaries

Gateway authentication, virtual-key scope, budgets and selected policy still
apply. Content inspection differs by input path:

| Input path | Current inspection boundary |
| --- | --- |
| Chat/Responses text | Conversation text is evaluated under the selected policies. Attachment text is extracted from the last message; earlier attachments are not generally re-extracted. |
| Inline PDF | Supported page/form text and font mappings are decoded within parser bounds. Font/image binary bytes are not treated as prose. PDF inspection-failure checks also visit attachments in earlier messages. |
| Inline image | Supported textual metadata is extracted; image pixels are not OCR-scanned by this extractor. |
| Dedicated image/audio/video operations | Eligible prompt, input and transcript text is evaluated when server multimodal guardrails are enabled. This does not provide automatic speech recognition of raw audio or inspection of video frames. |
| Uploaded file or `file_id` | Upload/management is not selected for file-content inspection by the LLM guardrail hook, and the extractor does not resolve provider file IDs. The `file` example validates lifecycle, not file-content guarding. |
| Remote file/image URL | A reference does not make the remote bytes available to the extractor; it does not fetch these URLs for content inspection. |

Dedicated multimodal operation evaluation requires **`GUARDRAILS_MULTIMODAL=true`**
on the server. This is an administrator setting, separate from the client's two
connection variables. It enables the implemented guard paths; it does not add
OCR, speech-to-text, video analysis or file resolution.

Output stream inspection is incremental. Text already delivered cannot be
recalled; cadence-based scanning can release chunks before evaluation. Use
nonstreaming inference or a separate buffering boundary when a full output
policy verdict must precede delivery. Redacting a final snapshot does not
retroactively sanitize earlier chunks.

An active enforcing policy that requires redaction inside a binary attachment
blocks before the provider when a safe rewrite is unavailable. Supply a
sanitized copy; changing the SDK or model does not remove that requirement.
Unsupported, encrypted or over-limit PDF inspection also follows the selected
policy execution mode. A clean image-only PDF can be valid while containing no
extractable text; that does not establish that its pixels are free of PII.

The [gateway guide](../../../deepintshield_server/docs/integrations/openai-sdk/all-providers.mdx#compatibility-performance-and-validation)
describes these boundaries and validation limits. Use approved synthetic inputs
for routine examples; use a suitable content-inspection workflow before relying
on uploaded file references or uninspected media for sensitive workloads.
