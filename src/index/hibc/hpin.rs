// In src/index/hibc/hpin.rs

use thiserror::Error;

/// A struct that encapsulates the logic for Hierarchical Prefix-ID Notation (HPIN).
#[derive(Debug, Clone)]
pub struct Hpin {
    alphabet: Vec<u8>, // Add this field
    alphabet_map: [u8; 256],
    n: usize,
    prefix_len: usize,
    powers: Vec<u64>,
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum HpinError {
    #[error("Word length must be {expected}, but got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("Character '{char}' not found in alphabet")]
    InvalidChar { char: char },
    #[error("Tail length 'm' must be between 1 and n-1, but got {actual}")]
    InvalidTailLength { actual: usize },
}

impl Hpin {
    pub fn new(alphabet: &[u8], n: usize, m: usize) -> Result<Self, HpinError> {
        if m == 0 || m >= n {
            return Err(HpinError::InvalidTailLength { actual: m });
        }
        let mut alphabet_map = [u8::MAX; 256];
        for (i, &byte_val) in alphabet.iter().enumerate() {
            alphabet_map[byte_val as usize] = i as u8;
        }
        let prefix_len = n - m;
        let k_u64 = alphabet.len() as u64;
        let powers = (0..prefix_len)
            .map(|p| k_u64.pow((prefix_len - 1 - p) as u32))
            .collect();
        Ok(Self { alphabet: alphabet.to_vec(), alphabet_map, n, prefix_len, powers })
    }

    pub fn parse<'a>(&self, word: &'a [u8]) -> Result<(u64, &'a [u8]), HpinError> {
        if word.len() != self.n {
            return Err(HpinError::InvalidLength { expected: self.n, actual: word.len() });
        }
        let (prefix, tail) = word.split_at(self.prefix_len);
        let pid = self.calculate_pid(prefix)?;
        Ok((pid, tail))
    }

    fn calculate_pid(&self, prefix: &[u8]) -> Result<u64, HpinError> {
        let mut pid: u64 = 0;
        for (i, &char_code) in prefix.iter().enumerate() {
            let ordinal = self.alphabet_map[char_code as usize];
            if ordinal == u8::MAX {
                return Err(HpinError::InvalidChar { char: char_code as char });
            }
            pid += (ordinal as u64) * self.powers[i];
        }
        Ok(pid)
    }

    pub fn n(&self) -> usize { self.n }
    pub fn m(&self) -> usize { self.n - self.prefix_len }
    pub fn alphabet(&self) -> &[u8] {
        &self.alphabet
    }
}
