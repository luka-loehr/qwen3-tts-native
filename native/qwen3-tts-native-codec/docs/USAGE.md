# Usage

## Consume the Rust library

Add the research crate as a path dependency while it remains in the playground:

```toml
[dependencies]
qwen3-tts-native-codec = { path = "../qwen3-tts-native-codec" }
```

Load the native library and decoder-only artifact once per process, then create
one owned session per active stream:

```rust
use qwen3_tts_native_codec::{
    CODEBOOKS, DecoderWeights, NativeCodecLibrary,
};
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let native = Arc::new(NativeCodecLibrary::load(Path::new(
        "build/native/libqwen3_tts_codec_cuda.so",
    ))?);
    let weights = DecoderWeights::open(Path::new(
        "speech_tokenizer/model.safetensors",
    ))?;

    let model = native
        .load_shared_model(0, &weights)
        .map_err(std::io::Error::other)?;
    assert_eq!(model.model_info().map_err(std::io::Error::other)?.tensor_count, 271);
    let mut stream = model.start_session().map_err(std::io::Error::other)?;

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

An indexed or mmap-backed runtime loader does not need to construct
`DecoderWeights`. Implement the public object-safe provider instead:

```rust
use qwen3_tts_native_codec::{DecoderWeightProvider, DecoderWeightTensor};

struct NativeArtifact {
    // mmap arena and verified tensor index
}

impl DecoderWeightProvider for NativeArtifact {
    fn decoder_tensor_names(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        // Return canonical decoder.* names from the verified index.
        todo!()
    }

    fn decoder_tensor(&self, name: &str) -> Option<DecoderWeightTensor<'_>> {
        // Return dtype, shape, and an arena slice valid through load_model.
        todo!("lookup {name}")
    }
}
```

`NativeCodecLibrary::load_shared_model(0, &artifact)` then uses the same C
callback path. The provider view is borrowed only for the duration of the
call; the shared native model owns its CUDA copy afterward.

## Process sessions concurrently

`NativeCodecModel` is `Send + Sync`, and each owned session is `Send + 'static`
but not `Sync`. Distinct sessions can be processed on scoped host threads:

```rust
# use qwen3_tts_native_codec::{CODEBOOKS, NativeCodecModel};
# use std::sync::Arc;
# fn example(model: &Arc<NativeCodecModel>) -> Result<(), Box<dyn std::error::Error>> {
let mut sessions = (0..3)
    .map(|_| model.start_session().map_err(std::io::Error::other))
    .collect::<Result<Vec<_>, _>>()?;
let packets = [
    [[1_u16; CODEBOOKS]],
    [[2_u16; CODEBOOKS]],
    [[3_u16; CODEBOOKS]],
];

std::thread::scope(|scope| {
    let workers = sessions
        .iter_mut()
        .zip(&packets)
        .map(|(session, packet)| {
            scope.spawn(move || session.process(packet, true))
        })
        .collect::<Vec<_>>();
    for worker in workers {
        let (pcm, _) = worker.join().expect("decoder worker did not panic")
            .map_err(|(status, message)| {
                std::io::Error::other(format!("decoder status {status}: {message}"))
            })?;
        assert_eq!(pcm.len(), 1_920);
    }
    Ok::<_, Box<dyn std::error::Error>>(())
})?;
# Ok(())
# }
```

Each worker uses an independent non-blocking CUDA stream and cuBLAS handle.
There is no global inference lock. Call `cancel()` to terminate one session;
siblings continue unaffected. `reset()` explicitly returns a session to fresh
state.

## Legacy array-order batch API

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

The handles must be independent. This compatibility API dispatches them in
array order and does not claim fused-batch acceleration or host concurrency.

## Run the real neural CLI gates

From `native/qwen3-tts-native-codec`:

```bash
LIB=build/native/libqwen3_tts_codec_cuda.so
BIN=target/release/qwen3-tts-native-codec
MODEL=/models/qwen3-tts-1.7b-voice-design-bf16-indexed/\
speech_tokenizer/model.safetensors
FIXTURE=../../benchmarks/fixtures/decoder-reference-bf16

# Full real neural PCM, lifecycle, short-final, and stale-tail parity.
$BIN neural-parity "$LIB" "$MODEL" "$FIXTURE"

# Official neural waveform checkpoints 6-13.
$BIN decoder-parity "$LIB" "$MODEL" "$FIXTURE"

# Independent B=3 and B=6 state-handle tests.
$BIN batch-parity "$LIB" "$MODEL" "$FIXTURE"

# Shared weights, B=1/B=3/B=6 interleaving, 20 concurrent stress rounds,
# reset/replay, cancel/drop, memory accounting, and official PCM.
$BIN shared-session-parity "$LIB" "$MODEL" "$FIXTURE" 20

# First real 80 ms packet without startup warmup.
$BIN neural-cold-start "$LIB" "$MODEL"

# Explicit startup warmup, then 20 warmups and 200 measurements per bucket.
$BIN neural-benchmark "$LIB" "$MODEL" 200

# Shared-model fresh-session TTFA and 200 measurements per packet bucket.
$BIN shared-neural-benchmark "$LIB" "$MODEL" 200
```

The legacy `parity` and `benchmark` commands exercise only a deterministic
state-machine fixture. They remain test utilities and must never be used as
neural latency or audio-quality evidence. All reportable model results come
from the commands above.

## Use the C ABI directly

Include `native/include/qwen3_tts_codec.h` and link or dynamically load
`libqwen3_tts_codec_cuda.so`. The shared-model order is:

1. `qwen3_tts_codec_shared_model_create_v1`
2. `qwen3_tts_codec_shared_model_load_v1`
3. `qwen3_tts_codec_shared_model_warmup_v1`
4. one `qwen3_tts_codec_session_create_v1` per active request
5. repeated `qwen3_tts_codec_session_process_packet_v1`
6. `qwen3_tts_codec_session_destroy_v1`
7. `qwen3_tts_codec_shared_model_destroy_v1`

Sessions retain the model internally, so a model owner may be released before
the final session; weights remain alive until the last reference is gone.
`qwen3_tts_codec_session_memory_info_v1` excludes shared weights by design.

The original ABI remains available:

1. `qwen3_tts_codec_create_v1`
2. `qwen3_tts_codec_load_model_v1`
3. `qwen3_tts_codec_warmup_v1`
4. repeated `qwen3_tts_codec_process_packet_v1`
5. `qwen3_tts_codec_destroy_v1`

Use `qwen3_tts_codec_reset_v1` only for an explicit stream replay/reuse. Check
every integer status and retain the error buffer until the call returns.
