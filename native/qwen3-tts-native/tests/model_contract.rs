use std::path::PathBuf;

use qwen3_tts_native::config::ModelConfig;
use qwen3_tts_native::prompt::{CodecFrame, TextMode, TextSource, VoiceDesignPrompt};
use qwen3_tts_native::tokenizer::Qwen2Tokenizer;
use qwen3_tts_native::weights::{SafeTensorProvider, WeightProvider};

fn model_directory() -> Option<PathBuf> {
    std::env::var_os("QWEN3_TTS_MODEL_DIR").map(PathBuf::from)
}

#[test]
fn pinned_voice_design_config_is_typed_and_validated() {
    let Some(directory) = model_directory() else {
        eprintln!("skipping real model test because QWEN3_TTS_MODEL_DIR is unset");
        return;
    };
    let config = ModelConfig::load(&directory.join("config.json")).unwrap();
    assert_eq!(config.talker_config.num_hidden_layers, 28);
    assert_eq!(
        config.talker_config.code_predictor_config.num_hidden_layers,
        5
    );
    assert_eq!(config.language_id("German").unwrap(), Some(2_053));
    assert!(config.language_id("Turkish").is_err());
}

#[test]
fn native_qwen_bpe_matches_official_oracle() {
    let Some(directory) = model_directory() else {
        eprintln!("skipping real model test because QWEN3_TTS_MODEL_DIR is unset");
        return;
    };
    let tokenizer = Qwen2Tokenizer::load(&directory).unwrap();

    let instruction = "<|im_start|>user\nA calm, warm adult male voice, measured and relaxed, with clear diction.<|im_end|>\n";
    assert_eq!(
        tokenizer.encode(instruction).unwrap(),
        [
            151_644, 872, 198, 32, 19_300, 11, 8_205, 6_683, 8_593, 7_743, 11, 16_878, 323, 30_367,
            11, 448, 2_797, 294, 2_479, 13, 151_645, 198,
        ]
    );

    let assistant = "<|im_start|>assistant\nGuten Abend. Heute testen wir die neue Stimme.<|im_end|>\n<|im_start|>assistant\n";
    assert_eq!(
        tokenizer.encode(assistant).unwrap(),
        [
            151_644, 77_091, 198, 38, 13_160, 3_680, 408, 13, 1_260, 1_070, 1_273, 268, 16_111,
            2_746, 38_383, 70_772, 2_660, 13, 151_645, 198, 151_644, 77_091, 198,
        ]
    );

    assert_eq!(
        tokenizer
            .encode("Français, Türkçe, italiano — déjà vu!")
            .unwrap(),
        [
            75_331, 3_131, 2_782, 11, 136_891, 11, 59_804, 1_959, 45_839, 32_514, 0
        ]
    );
}

#[test]
fn streaming_prompt_matches_official_embedding_order() {
    let Some(directory) = model_directory() else {
        eprintln!("skipping real model test because QWEN3_TTS_MODEL_DIR is unset");
        return;
    };
    let tokenizer = Qwen2Tokenizer::load(&directory).unwrap();
    let config = ModelConfig::load(&directory.join("config.json")).unwrap();
    let prompt = VoiceDesignPrompt::tokenize(
        &tokenizer,
        &config,
        "Guten Abend. Heute testen wir die neue Stimme.",
        "A calm, warm adult male voice, measured and relaxed, with clear diction.",
        "German",
        TextMode::Streaming,
    )
    .unwrap();

    assert_eq!(
        prompt.prefill[..prompt.instruction_ids.len()]
            .iter()
            .map(|step| step.text)
            .collect::<Vec<_>>(),
        prompt
            .instruction_ids
            .iter()
            .copied()
            .map(TextSource::Token)
            .map(Some)
            .collect::<Vec<_>>()
    );

    let codec_steps = prompt
        .prefill
        .iter()
        .filter_map(|step| step.codec)
        .collect::<Vec<_>>();
    assert_eq!(codec_steps, [2_154, 2_156, 2_053, 2_157, 2_148, 2_149]);
    assert_eq!(prompt.trailing_text.last(), Some(&TextSource::TtsEos));
    assert_eq!(
        prompt.prefill.last().unwrap().text,
        Some(TextSource::Token(prompt.assistant_ids[3]))
    );
}

#[test]
fn frame_order_and_eos_contract_are_explicit() {
    let Some(directory) = model_directory() else {
        eprintln!("skipping real model test because QWEN3_TTS_MODEL_DIR is unset");
        return;
    };
    let config = ModelConfig::load(&directory.join("config.json")).unwrap();
    let residual = (101_u32..116).collect::<Vec<_>>();
    let frame = CodecFrame::from_predictor(77, &residual).unwrap();
    assert_eq!(frame.0[0], 77);
    assert_eq!(&frame.0[1..], residual);
    assert!(!frame.is_eos(&config));

    let residual_eos = CodecFrame::from_predictor(77, &[2_150; 15]).unwrap();
    assert!(!residual_eos.is_eos(&config));
    let semantic_eos = CodecFrame::from_predictor(2_150, &[77; 15]).unwrap();
    assert!(semantic_eos.is_eos(&config));
}

#[test]
fn real_checkpoint_provider_exposes_talker_and_predictor_weights() {
    let Some(directory) = model_directory() else {
        eprintln!("skipping real model test because QWEN3_TTS_MODEL_DIR is unset");
        return;
    };
    let provider = SafeTensorProvider::open(&directory.join("model.safetensors")).unwrap();
    assert_eq!(provider.tensor_names().unwrap().len(), 404);
    provider
        .expect_bf16("talker.model.text_embedding.weight", &[151_936, 2_048])
        .unwrap();
    provider
        .expect_bf16(
            "talker.model.layers.0.self_attn.q_proj.weight",
            &[2_048, 2_048],
        )
        .unwrap();
    provider
        .expect_bf16("talker.code_predictor.lm_head.14.weight", &[2_048, 1_024])
        .unwrap();
}
