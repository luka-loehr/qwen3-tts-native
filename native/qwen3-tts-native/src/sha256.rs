use std::fmt::Write as _;
use std::io::{self, Read};

const INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    block_len: usize,
    bytes_seen: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self {
            state: INITIAL_STATE,
            block: [0; 64],
            block_len: 0,
            bytes_seen: 0,
        }
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, mut input: &[u8]) {
        self.bytes_seen = self
            .bytes_seen
            .checked_add(input.len() as u64)
            .expect("SHA-256 inputs longer than u64::MAX bytes are unsupported");

        if self.block_len != 0 {
            let copied = (64 - self.block_len).min(input.len());
            self.block[self.block_len..self.block_len + copied].copy_from_slice(&input[..copied]);
            self.block_len += copied;
            input = &input[copied..];
            if self.block_len == 64 {
                let block = self.block;
                self.compress(&block);
                self.block_len = 0;
            } else {
                return;
            }
        }

        let mut chunks = input.chunks_exact(64);
        for chunk in &mut chunks {
            let block: &[u8; 64] = chunk.try_into().expect("exact chunk size");
            self.compress(block);
        }

        let remainder = chunks.remainder();
        self.block[..remainder.len()].copy_from_slice(remainder);
        self.block_len = remainder.len();
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let message_bits = self
            .bytes_seen
            .checked_mul(8)
            .expect("SHA-256 inputs longer than 2^61 - 1 bytes are unsupported");
        self.block[self.block_len] = 0x80;
        self.block_len += 1;

        if self.block_len > 56 {
            self.block[self.block_len..].fill(0);
            let block = self.block;
            self.compress(&block);
            self.block = [0; 64];
        } else {
            self.block[self.block_len..56].fill(0);
        }

        self.block[56..64].copy_from_slice(&message_bits.to_be_bytes());
        let block = self.block;
        self.compress(&block);

        let mut digest = [0_u8; 32];
        for (word, destination) in self.state.iter().zip(digest.chunks_exact_mut(4)) {
            destination.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut schedule = [0_u32; 64];
        for (index, bytes) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes(bytes.try_into().expect("four bytes"));
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }

        for (state, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(value);
        }
    }
}

pub fn digest_bytes(input: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finalize()
}

pub fn digest_reader(reader: &mut impl Read, buffer_bytes: usize) -> io::Result<([u8; 32], u64)> {
    if buffer_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SHA-256 buffer size must be greater than zero",
        ));
    }

    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; buffer_bytes];
    let mut bytes = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| io::Error::other("hashed byte count overflowed"))?;
        hasher.update(&buffer[..read]);
    }
    Ok((hasher.finalize(), bytes))
}

pub fn to_hex(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{Sha256, digest_bytes, digest_reader, to_hex};
    use std::io::Cursor;

    #[test]
    fn matches_fips_vectors() {
        assert_eq!(
            to_hex(&digest_bytes(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            to_hex(&digest_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            to_hex(&digest_bytes(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn incremental_boundaries_match_one_shot() {
        let input = (0_u32..20_000)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let expected = digest_bytes(&input);

        for chunk_bytes in [1, 7, 55, 56, 63, 64, 65, 4096] {
            let mut hasher = Sha256::new();
            for chunk in input.chunks(chunk_bytes) {
                hasher.update(chunk);
            }
            assert_eq!(hasher.finalize(), expected, "chunk size {chunk_bytes}");
        }
    }

    #[test]
    fn reader_is_bounded_and_counts_bytes() {
        let input = vec![0x5a; 1_000_003];
        let (digest, bytes) = digest_reader(&mut Cursor::new(&input), 8191).expect("hash reader");
        assert_eq!(bytes, input.len() as u64);
        assert_eq!(digest, digest_bytes(&input));
    }

    #[test]
    fn reader_rejects_zero_sized_buffer() {
        let error = digest_reader(&mut Cursor::new([]), 0).expect_err("must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }
}
