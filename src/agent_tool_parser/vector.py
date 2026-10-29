# Copyright (c) 2026 InBoost Team
# SPDX-License-Identifier: MIT

"""Hardware architecture and vector instruction detection (AVX2, SSE4.2, NEON, Scalar).

Provides runtime SIMD capability detection and safe fallbacks for high-throughput
agent parser operations.
"""

from __future__ import annotations

import os
import platform
from enum import Enum


class CpuVectorEngine(str, Enum):
    """CPU vector instruction engine detected at runtime."""

    Avx2_256 = "avx2_256"
    Sse42_128 = "sse42_128"
    Neon_128 = "neon_128"
    Scalar_Fallback = "scalar_fallback"

    def __str__(self) -> str:
        return self.value


def _detect_cpu_flags_fallback() -> CpuVectorEngine:
    """Fallback detection by inspecting CPU architecture and flags when running pure-Python."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        # On Linux, inspect /proc/cpuinfo
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    if "avx2" in content:
                        return CpuVectorEngine.Avx2_256
                    if "sse4_2" in content or "sse4.2" in content:
                        return CpuVectorEngine.Sse42_128
            except Exception:
                pass
        return CpuVectorEngine.Scalar_Fallback

    if machine in ("aarch64", "arm64"):
        return CpuVectorEngine.Neon_128

    return CpuVectorEngine.Scalar_Fallback


def detect_vector_engine() -> CpuVectorEngine:
    """Detects active CPU vector instruction engine at runtime."""
    # Try accelerated native extension if available and not explicitly disabled
    if os.getenv("AGENT_TOOL_PARSER_NO_EXT", "0") not in ("1", "true", "True"):
        try:
            from agent_tool_parser._accelerated import (  # type: ignore[import-untyped,import-not-found]
                detect_vector_engine as _native_detect,
            )

            raw = _native_detect()
            return CpuVectorEngine(raw)
        except (ImportError, AttributeError, ValueError):
            pass

    return _detect_cpu_flags_fallback()
