use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use fancy_regex::Regex;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

const PRETOKENIZE_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
}

#[derive(Debug)]
pub struct Qwen2Tokenizer {
    pattern: Regex,
    vocabulary: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    added_tokens: Vec<(String, u32)>,
}

impl Qwen2Tokenizer {
    pub fn load(model_directory: &Path) -> Result<Self> {
        let vocabulary_path = model_directory.join("vocab.json");
        let merges_path = model_directory.join("merges.txt");
        let config_path = model_directory.join("tokenizer_config.json");

        let vocabulary: HashMap<String, u32> = serde_json::from_slice(
            &fs::read(&vocabulary_path)
                .with_context(|| format!("failed to read {}", vocabulary_path.display()))?,
        )
        .with_context(|| format!("invalid vocabulary {}", vocabulary_path.display()))?;

        let merges = fs::read_to_string(&merges_path)
            .with_context(|| format!("failed to read {}", merges_path.display()))?;
        let mut merge_ranks = HashMap::new();
        for (rank, line) in merges
            .lines()
            .filter(|line| !line.starts_with('#'))
            .enumerate()
        {
            let mut symbols = line.split_whitespace();
            let left = symbols.next().context("merge is missing its left symbol")?;
            let right = symbols
                .next()
                .context("merge is missing its right symbol")?;
            if symbols.next().is_some() {
                bail!("merge line contains more than two symbols: {line:?}");
            }
            merge_ranks.insert((left.to_owned(), right.to_owned()), rank);
        }

        let config: TokenizerConfig = serde_json::from_slice(
            &fs::read(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?,
        )
        .with_context(|| format!("invalid tokenizer config {}", config_path.display()))?;
        let mut added_tokens = config
            .added_tokens_decoder
            .into_iter()
            .map(|(id, token)| {
                let id = id
                    .parse::<u32>()
                    .with_context(|| format!("invalid added-token ID {id:?}"))?;
                Ok((token.content, id))
            })
            .collect::<Result<Vec<_>>>()?;
        added_tokens.sort_unstable_by(|left, right| {
            right
                .0
                .len()
                .cmp(&left.0.len())
                .then_with(|| left.1.cmp(&right.1))
        });

        Ok(Self {
            pattern: Regex::new(PRETOKENIZE_REGEX).context("invalid Qwen2 pre-tokenizer regex")?,
            vocabulary,
            merge_ranks,
            byte_encoder: build_byte_encoder(),
            added_tokens,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = text.nfc().collect::<String>();
        let mut ids = Vec::new();
        let mut cursor = 0;

        while cursor < normalized.len() {
            if let Some((offset, token, id)) = self.next_added_token(&normalized[cursor..]) {
                if offset > 0 {
                    self.encode_ordinary(&normalized[cursor..cursor + offset], &mut ids)?;
                }
                ids.push(id);
                cursor += offset + token.len();
            } else {
                self.encode_ordinary(&normalized[cursor..], &mut ids)?;
                cursor = normalized.len();
            }
        }
        Ok(ids)
    }

    fn next_added_token<'a>(&'a self, text: &str) -> Option<(usize, &'a str, u32)> {
        self.added_tokens
            .iter()
            .filter_map(|(token, id)| text.find(token).map(|offset| (offset, token.as_str(), *id)))
            .min_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| right.1.len().cmp(&left.1.len()))
                    .then_with(|| left.2.cmp(&right.2))
            })
    }

    fn encode_ordinary(&self, text: &str, output: &mut Vec<u32>) -> Result<()> {
        for result in self.pattern.find_iter(text) {
            let piece = result.context("Qwen2 pre-tokenizer failed")?.as_str();
            let encoded = piece
                .as_bytes()
                .iter()
                .map(|byte| self.byte_encoder[*byte as usize])
                .collect::<String>();
            for token in self.byte_pair_encode(&encoded) {
                let id = self.vocabulary.get(&token).copied().with_context(|| {
                    format!("BPE token {token:?} is missing from the vocabulary")
                })?;
                output.push(id);
            }
        }
        Ok(())
    }

    fn byte_pair_encode(&self, token: &str) -> Vec<String> {
        let mut symbols = token
            .chars()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let best = symbols
                .windows(2)
                .filter_map(|pair| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .copied()
                        .map(|rank| (rank, pair[0].as_str(), pair[1].as_str()))
                })
                .min_by_key(|(rank, _, _)| *rank);
            let Some((_, left, right)) = best else {
                break;
            };
            let left = left.to_owned();
            let right = right.to_owned();
            let mut merged = Vec::with_capacity(symbols.len());
            let mut index = 0;
            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    merged.push(format!("{left}{right}"));
                    index += 2;
                } else {
                    merged.push(symbols[index].clone());
                    index += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }
}

fn build_byte_encoder() -> [char; 256] {
    let mut values = Vec::with_capacity(256);
    values.extend(33_u16..=126);
    values.extend(161_u16..=172);
    values.extend(174_u16..=255);

    let mut present = [false; 256];
    for value in &values {
        present[*value as usize] = true;
    }
    let mut mapped = values.clone();
    let mut offset = 0_u16;
    for byte in 0_u16..=255 {
        if !present[byte as usize] {
            values.push(byte);
            mapped.push(256 + offset);
            offset += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    for (byte, codepoint) in values.into_iter().zip(mapped) {
        encoder[byte as usize] =
            char::from_u32(codepoint as u32).expect("valid byte encoder codepoint");
    }
    encoder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_encoder_is_bijective() {
        let encoder = build_byte_encoder();
        let mut values = encoder.to_vec();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), 256);
    }
}
