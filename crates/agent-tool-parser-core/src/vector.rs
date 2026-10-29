// Copyright (c) 2026 InBoost Team
// SPDX-License-Identifier: MIT

//! Safe hardware architecture and vector instruction detection (AVX2, SSE4.2, NEON, Scalar).
//! Provides runtime SIMD capability detection and high-performance vector scanning utilities
//! via `memchr` without risking SIGILL illegal instruction crashes on virtualized or legacy CPUs.

use serde::{Deserialize, Serialize};

/// CPU vector instruction engine detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum CpuVectorEngine {
    /// x86_64: 256-bit AVX2 + FMA vector acceleration.
    Avx2_256,
    /// x86_64: 128-bit SSE4.2 vector acceleration.
    Sse42_128,
    /// aarch64: 128-bit ARM Advanced SIMD (NEON).
    Neon_128,
    /// Portable scalar fallback (works across WASM, older x86/ARM, RISC-V).
    Scalar_Fallback,
}

impl CpuVectorEngine {
    /// String representation of the detected vector engine.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Avx2_256 => "avx2_256",
            Self::Sse42_128 => "sse42_128",
            Self::Neon_128 => "neon_128",
            Self::Scalar_Fallback => "scalar_fallback",
        }
    }
}

impl std::fmt::Display for CpuVectorEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Dynamically detects the active CPU architecture and available SIMD vector features at runtime.
///
/// Guaranteed to never panic or generate illegal instructions (SIGILL) even on
/// virtualized cloud runners (Docker, GitHub Actions, microVMs) or legacy processors.
pub fn detect_vector_engine() -> CpuVectorEngine {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return CpuVectorEngine::Avx2_256;
        }
        if std::is_x86_feature_detected!("sse4.2") {
            return CpuVectorEngine::Sse42_128;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return CpuVectorEngine::Neon_128;
        }
    }

    CpuVectorEngine::Scalar_Fallback
}

/// Searches for the first occurrence of a multi-byte needle in haystack using SIMD acceleration.
#[inline]
pub fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    memchr::memmem::find(haystack, needle)
}

/// Finds the first occurrence of a single byte in haystack using SIMD acceleration.
#[inline]
pub fn find_byte(haystack: &[u8], byte: u8) -> Option<usize> {
    memchr::memchr(byte, haystack)
}

/// Finds the first occurrence of either of two bytes in haystack using SIMD acceleration.
#[inline]
pub fn find_byte2(haystack: &[u8], b1: u8, b2: u8) -> Option<usize> {
    memchr::memchr2(b1, b2, haystack)
}

/// Finds the first occurrence of any of three bytes in haystack using SIMD acceleration.
#[inline]
pub fn find_byte3(haystack: &[u8], b1: u8, b2: u8, b3: u8) -> Option<usize> {
    memchr::memchr3(b1, b2, b3, haystack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_vector_engine_returns_valid_variant() {
        let engine = detect_vector_engine();
        let name = engine.as_str();
        assert!(!name.is_empty());
        assert_eq!(format!("{}", engine), name);

        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                engine == CpuVectorEngine::Avx2_256
                    || engine == CpuVectorEngine::Sse42_128
                    || engine == CpuVectorEngine::Scalar_Fallback
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(
                engine == CpuVectorEngine::Neon_128 || engine == CpuVectorEngine::Scalar_Fallback
            );
        }
    }

    #[test]
    fn test_simd_find_subsequence() {
        let haystack = b"Hello, <tool_call>{\"name\": \"bash\"}</tool_call>!";
        let pos = find_subsequence(haystack, b"<tool_call>");
        assert_eq!(pos, Some(7));

        let not_found = find_subsequence(haystack, b"<missing>");
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_simd_find_byte_and_variants() {
        let text = b"foo:bar,baz{qux}";
        assert_eq!(find_byte(text, b':'), Some(3));
        assert_eq!(find_byte2(text, b',', b'{'), Some(7));
        assert_eq!(find_byte3(text, b'}', b'{', b','), Some(7));
        assert_eq!(find_byte(text, b'Z'), None);
    }
}
