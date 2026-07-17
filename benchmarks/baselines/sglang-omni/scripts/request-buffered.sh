#!/usr/bin/env bash
set -Eeuo pipefail

url="${URL:-http://127.0.0.1:8000/v1/audio/speech}"
output="${1:-/tmp/sglang-omni-buffered.wav}"
text="${TEXT:-Guten Morgen. Heute messen wir die Audioerzeugung reproduzierbar und ohne übertriebene Betonung.}"
instructions="${INSTRUCTIONS:-A calm adult male teacher with a warm low register, measured pacing, clear Standard German articulation, and restrained emotion.}"
language="${LANGUAGE_HINT:-German}"
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
    response_format: "wav",
    stream: false,
    max_new_tokens: 256,
    temperature: 0.9,
    top_p: 1.0,
    top_k: 50,
    repetition_penalty: 1.05,
    seed: $seed
  }' \
  | curl --fail-with-body --silent --show-error \
      --request POST \
      --header 'Content-Type: application/json' \
      --data-binary @- \
      --output "${output}" \
      "${url}"

echo "Wrote ${output}"
