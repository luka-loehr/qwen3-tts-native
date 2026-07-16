# Usage

## Consume the Rust library

Add the research crate as a path dependency while it remains in the playground:

```toml
[dependencies]
qwen3-tts-native-codec = { path = "../qwen3-tts-native-codec" }
```

Load the native library and decoder-only artifact once per process, then create
one state handle per active stream:

```rust
use qwen3_tts_native_codec::{
    CODEBOOKS, DecoderWeights, NativeCodecLibrary,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let native = NativeCodecLibrary::load(Path::new(
        "build/native/libqwen3_tts_codec_cuda.so",
    ))?;
    let weights = DecoderWeights::open(Path::new(
        "speech_tokenizer/model.safetensors",
    ))?;

    let mut stream = native.create_codec(0).map_err(std::io::Error::other)?;
    let model = stream.load_model(&weights).map_err(std::io::Error::other)?;
    assert_eq!(model.tensor_count, 271);
    stream.warmup().map_err(std::io::Error::other)?;

    let frames: [[u16; CODEBOOKS]; 1] = [[0; CODEBOOKS]];
    let (pcm, result) = stream
        .process(&frames, true)
        .map_err(|(status, message)| {
            std::io::Error::other(format!("decoder status {status}: {message}"))
        })?;
    assert_eq!(pcm.len(), 1_920);
    assert_eq!(result.sample_count, 1_920);
    Ok(())
}
```

`DecoderWeights` accepts the canonical decoder-only BF16 artifact and the
original F32 speech-tokenizer checkpoint. The native execution layer reports
source and device bytes independently.

## Decode independent streams through the batch API

The batch method accepts mutable references to distinct state handles, packet
slices, and final flags of equal length:

```rust
# use qwen3_tts_native_codec::{CODEBOOKS, NativeCodec, NativeCodecLibrary};
# fn example(
#     native: &NativeCodecLibrary,
#     left: &mut NativeCodec<'_>,
#     right: &mut NativeCodec<'_>,
# ) -> Result<(), Box<dyn std::error::Error>> {
let left_frames = [[1_u16; CODEBOOKS]];
let right_frames = [[2_u16; CODEBOOKS]; 3];
let mut streams = [left, right];
let packets: [&[[u16; CODEBOOKS]]; 2] = [&left_frames, &right_frames];
let output = native
    .process_batch(&mut streams, &packets, &[false, true])
    .map_err(|(status, message)| {
        std::io::Error::other(format!("batch status {status}: {message}"))
    })?;
assert_eq!(output[0].0.len(), 1_920);
assert_eq!(output[1].0.len(), 5_760);
# Ok(())
# }
```

The handles must be independent. The current implementation dispatches them in
array order and does not claim fused-batch acceleration.

## Run the real neural CLI gates

From `native/qwen3-tts-native-codec`:

```bash
LIB=build/native/libqwen3_tts_codec_cuda.so
BIN=target/release/qwen3-tts-native-codec
MODEL=/home/administrator/codex-playground-artifacts/\
qwen3-tts-1.7b-voice-design-bf16-indexed/speech_tokenizer/model.safetensors
FIXTURE=../../benchmarks/fixtures/decoder-reference-bf16

# Full real neural PCM, lifecycle, short-final, and stale-tail parity.
$BIN neural-parity "$LIB" "$MODEL" "$FIXTURE"

# Official neural waveform checkpoints 6-13.
$BIN decoder-parity "$LIB" "$MODEL" "$FIXTURE"

# Independent B=3 and B=6 state-handle tests.
$BIN batch-parity "$LIB" "$MODEL" "$FIXTURE"

# First real 80 ms packet without startup warmup.
$BIN neural-cold-start "$LIB" "$MODEL"

# Explicit startup warmup, then 20 warmups and 200 measurements per bucket.
$BIN neural-benchmark "$LIB" "$MODEL" 200
```

The legacy `parity` and `benchmark` commands exercise only a deterministic
state-machine fixture. They remain test utilities and must never be used as
neural latency or audio-quality evidence. All reportable model results come
from the commands above.

## Use the C ABI directly

Include `native/include/qwen3_tts_codec.h` and link or dynamically load
`libqwen3_tts_codec_cuda.so`. The required order is:

1. `qwen3_tts_codec_create_v1`
2. `qwen3_tts_codec_load_model_v1`
3. `qwen3_tts_codec_warmup_v1`
4. repeated `qwen3_tts_codec_process_packet_v1`
5. `qwen3_tts_codec_destroy_v1`

Use `qwen3_tts_codec_reset_v1` only for an explicit stream replay/reuse. Check
every integer status and retain the error buffer until the call returns.
