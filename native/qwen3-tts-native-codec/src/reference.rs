use crate::ffi::{CODEBOOKS, SAMPLES_PER_FRAME};

pub struct ReferenceState {
    preconv: [i32; 2],
    transpose_tails: [Vec<i32>; 4],
}

impl ReferenceState {
    pub fn new() -> Self {
        Self {
            preconv: [0; 2],
            transpose_tails: [vec![0; 8], vec![0; 5], vec![0; 4], vec![0; 3]],
        }
    }

    pub fn process(&mut self, frames: &[[u16; CODEBOOKS]]) -> Vec<i16> {
        let mut values = Vec::with_capacity(frames.len());
        for frame in frames {
            let weighted_sum = frame
                .iter()
                .enumerate()
                .map(|(index, code)| i32::from(code & 2047) * (index as i32 + 1))
                .sum::<i32>();
            let centered = (weighted_sum & 4095) - 2048;
            let filtered = (3 * centered + 2 * self.preconv[0] - self.preconv[1]) / 4;
            self.preconv[1] = self.preconv[0];
            self.preconv[0] = centered;
            values.push(filtered);
        }

        values = repeat(&values, 2);
        values = repeat(&values, 2);
        for (stage, stride) in [8, 5, 4, 3].into_iter().enumerate() {
            values = transpose_overlap(&values, stride, &mut self.transpose_tails[stage]);
        }
        debug_assert_eq!(values.len(), frames.len() * SAMPLES_PER_FRAME);
        values
            .into_iter()
            .map(|value| (value * 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
            .collect()
    }
}

fn repeat(input: &[i32], count: usize) -> Vec<i32> {
    let mut output = Vec::with_capacity(input.len() * count);
    for value in input {
        output.extend(std::iter::repeat_n(*value, count));
    }
    output
}

fn transpose_overlap(input: &[i32], stride: usize, prior_tail: &mut [i32]) -> Vec<i32> {
    assert!(!input.is_empty());
    assert_eq!(prior_tail.len(), stride);
    let mut output = Vec::with_capacity(input.len() * stride);
    for (index, value) in input.iter().enumerate() {
        for tail_value in prior_tail.iter() {
            let previous = if index == 0 {
                *tail_value
            } else {
                input[index - 1]
            };
            output.push((3 * value + previous) / 4);
        }
    }
    prior_tail.fill(*input.last().expect("input is not empty"));
    output
}
