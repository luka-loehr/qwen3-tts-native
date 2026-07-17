#include "qwen3_tts_codec.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using BeginFn = int32_t (*)(
    Qwen3TtsCodecSessionV1*,
    const Qwen3TtsCodecDevicePacketBeginV2*,
    Qwen3TtsCodecDevicePacketBeginResultV2*,
    char*,
    size_t
);

using FinishFn = int32_t (*)(
    Qwen3TtsCodecSessionV1*,
    const Qwen3TtsCodecDevicePacketFinishV2*,
    char*,
    size_t
);

static_assert(QWEN3_TTS_CODEC_ABI_VERSION_V1 == 1);
static_assert(QWEN3_TTS_CODEC_ABI_VERSION_V2 == 2);
static_assert(std::is_standard_layout_v<Qwen3TtsCodecDevicePacketBeginV2>);
static_assert(std::is_standard_layout_v<Qwen3TtsCodecDevicePacketBeginResultV2>);
static_assert(std::is_standard_layout_v<Qwen3TtsCodecDevicePacketFinishV2>);
static_assert(offsetof(Qwen3TtsCodecDevicePacketBeginV2, struct_size) == 0);
static_assert(offsetof(Qwen3TtsCodecDevicePacketBeginResultV2, struct_size) == 0);
static_assert(offsetof(Qwen3TtsCodecDevicePacketFinishV2, struct_size) == 0);
static_assert(std::is_same_v<
              decltype(&qwen3_tts_codec_session_process_device_packet_begin_v2),
              BeginFn>);
static_assert(std::is_same_v<
              decltype(&qwen3_tts_codec_session_process_device_packet_finish_v2),
              FinishFn>);

}  // namespace

int main() {
    Qwen3TtsCodecDevicePacketBeginV2 begin{};
    begin.struct_size = sizeof(begin);
    Qwen3TtsCodecDevicePacketBeginResultV2 output{};
    output.struct_size = sizeof(output);
    Qwen3TtsCodecDevicePacketFinishV2 finish{};
    finish.struct_size = sizeof(finish);
    return begin.reserved == 0 && begin.reserved_2 == 0 &&
                   begin.reserved_3 == 0 && output.reserved == 0 &&
                   output.reserved_2 == 0 && finish.reserved == 0 &&
                   finish.reserved_2 == 0 && finish.reserved_3 == 0
               ? 0
               : 1;
}
