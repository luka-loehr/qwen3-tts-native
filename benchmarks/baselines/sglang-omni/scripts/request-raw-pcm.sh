#!/usr/bin/env bash
set -Eeuo pipefail

url="${URL:-http://127.0.0.1:8000/v1/audio/speech}"
output="${1:-/tmp/sglang-omni-raw.pcm}"
headers="${output}.headers"
text="${TEXT:-The first byte timestamp must be measured independently from the final response timestamp.}"
instructions="${INSTRUCTIONS:-A calm adult male engineer with a low natural pitch, deliberate pacing, clear international English, and understated confidence.}"
language="${LANGUAGE_HINT:-English}"
seed="${SEED:-12345}"

jq -n \
  --arg input "${text}" \
  --arg instructions "${instructions}" \
  --arg language "${language}" \
  --argjson seed "${seed}" \
  '{
    input: $input,
    task_type: "VoiceDesign",
    instructions: $instructions,
    language: $language,
    response_format: "pcm",
    stream: true,
    stream_format: "audio",
    max_new_tokens: 256,
    temperature: 0.9,
    top_p: 1.0,
    top_k: 50,
    repetition_penalty: 1.05,
    seed: $seed
  }' \
  | curl --fail-with-body --silent --show-error --no-buffer \
      --request POST \
      --header 'Content-Type: application/json' \
      --data-binary @- \
      --dump-header "${headers}" \
      --output "${output}" \
      "${url}"

echo "Wrote ${output} and ${headers}"
