"""
model_recommender.py  v3
Hardware-aware GGUF LLM recommender for llama.cpp.

Memory model covers all four costs that matter in practice:
  1. Transformer weights      M = P × b / 8  (P=params, b=bits per weight)
  2. KV cache                 2 × layers × kv_heads × head_dim × ctx_tokens × dtype_bytes
  3. CPU compute buffer       llama.cpp ggml graph: num_heads × n_batch × ctx × 4 bytes
                              (only on CPU; GPU/Metal handles this in device memory at ~0 host cost)
  4. Linux page-cache         measured from /proc/meminfo; mmap'd model pages compete with
                              kernel page cache — not relevant on GPU or Apple Unified Memory
  5. Tokenizer process        flat 200 MB (llama.cpp binary + vocab + sentencepiece runtime)

Device type influences both the memory budget AND the practical model ceiling:
  • GPU VRAM      — high bandwidth (300–3000 GB/s), fast; recommend largest that fits
  • Apple Silicon — unified memory with Metal; very high bandwidth for M-series
  • CPU-only      — bandwidth-limited (25–70 GB/s); flags models too slow to be useful
                    (<1.5 t/s) and further penalises the absence of AVX2
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    sys.exit("psutil is required:  pip install psutil")


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL TUNABLES
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONTEXT_K     = 4        # default inference context (K tokens; 4 = 4096)
DEFAULT_KV_BITS       = 16       # KV-cache dtype: 16 = fp16 (default), 8 = int8 q8_0
MIN_PRACTICAL_TPS     = 1.5      # tokens/sec below which CPU inference is flagged
TOKENIZER_OVERHEAD_GB = 0.20     # llama.cpp process + vocab + sentencepiece runtime
LLAMA_N_BATCH         = 512      # llama.cpp default --n-batch (used for compute-buf estimate)

# Per-source absolute headroom that counts as "comfortable"
# (fraction-based margins explode on large machines and starve small ones)
_COMFORT_HEADROOM_GB = {
    "GPU VRAM":              0.5,
    "Apple Unified Memory":  1.5,
    "System RAM (CPU-only)": 2.0,
}
COMFORT_RATIO = 0.12

BROWSER_OVERHEAD_GB = 2.5

# Fraction of total memory that the model process can actually use
_GPU_EFFICIENCY   = 0.90   # CUDA context, driver, NCCL buffers
_RAM_EFFICIENCY   = 0.65   # OS, page tables, other apps, swap hysteresis
_APPLE_EFFICIENCY = 0.78   # macOS metal driver + OS reserved pages


# ──────────────────────────────────────────────────────────────────────────────
# 1.  QUANT CATALOGUE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Quant:
    name:         str
    bpw:          float   # bits-per-weight (source of truth)
    tier:         str     # ULTRA_FAST | FAST | BALANCED | QUALITY | REFERENCE
    quality_note: str
    factor:       float = field(init=False)   # bytes per param = bpw / 8

    def __post_init__(self) -> None:
        self.factor = self.bpw / 8.0


QUANTS: list[Quant] = [
    # ── ultra-low ──────────────────────────────────────────────────────────────
    Quant("IQ1_S",   1.56, "ULTRA_FAST", "1-bit iMatrix; severe degradation, last resort"),
    Quant("IQ2_XXS", 2.06, "ULTRA_FAST", "2-bit iMatrix; smallest practical GGUF"),
    Quant("IQ2_XS",  2.31, "ULTRA_FAST", "2-bit iMatrix XS"),
    Quant("IQ2_M",   2.57, "ULTRA_FAST", "2-bit iMatrix M; approaching Q3 quality"),
    Quant("Q2_K",    2.63, "ULTRA_FAST", "2-bit k-quant; fast but noticeably lossy"),
    # ── low ────────────────────────────────────────────────────────────────────
    Quant("IQ3_XXS", 3.06, "FAST",       "3-bit iMatrix XXS"),
    Quant("IQ3_XS",  3.30, "FAST",       "3-bit iMatrix XS"),
    Quant("Q3_K_S",  3.50, "FAST",       "3-bit k-quant small"),
    Quant("Q3_K_M",  3.91, "FAST",       "3-bit k-quant medium; good speed/size ratio"),
    Quant("Q3_K_L",  4.27, "FAST",       "3-bit k-quant large"),
    # ── balanced ───────────────────────────────────────────────────────────────
    Quant("IQ4_XS",  4.25, "BALANCED",   "4-bit iMatrix XS; best size/quality among 4-bit"),
    Quant("IQ4_NL",  4.50, "BALANCED",   "4-bit iMatrix NL"),
    Quant("Q4_0",    4.55, "BALANCED",   "4-bit legacy; slightly behind Q4_K_M"),
    Quant("Q4_K_S",  4.37, "BALANCED",   "4-bit k-quant small"),
    Quant("Q4_K_M",  4.85, "BALANCED",   "4-bit k-quant medium; community sweet-spot"),
    Quant("Q4_K_L",  4.90, "BALANCED",   "4-bit k-quant large"),
    # ── quality ────────────────────────────────────────────────────────────────
    Quant("Q5_K_S",  5.54, "QUALITY",    "5-bit k-quant small"),
    Quant("Q5_K_M",  5.68, "QUALITY",    "5-bit k-quant medium; near-lossless"),
    Quant("Q6_K",    6.57, "QUALITY",    "6-bit k-quant; very close to FP16"),
    Quant("Q8_0",    8.50, "QUALITY",    "8-bit; indistinguishable from FP16 in practice"),
    Quant("F16",    16.00, "REFERENCE",  "Full FP16; reference quality, enormous VRAM cost"),
]

QUANT_BY_NAME: dict[str, Quant] = {q.name: q for q in QUANTS}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  MODEL CATALOGUE  – architecture required for exact memory math
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelArch:
    """
    Transformer dimensions needed to compute KV-cache and CPU compute-buffer sizes.

    KV-cache formula (llama.cpp, fp16 default):
        bytes = 2 (K+V) × layers × kv_heads × head_dim × n_ctx × dtype_bytes

    CPU compute-buffer dominant term (attention score matrix before softmax):
        bytes = num_heads × n_batch × n_ctx × 4  (float32)
        where num_heads = hidden_size // head_dim
    """
    num_layers:   int
    num_kv_heads: int    # GQA KV heads (< num_heads for modern models)
    head_dim:     int    # Q/K/V head dimension
    hidden_size:  int    # embedding / residual stream dimension
    vocab_size:   int    # for reference; embeddings are part of weight count

    @property
    def num_heads(self) -> int:
        return self.hidden_size // self.head_dim


@dataclass
class ModelFamily:
    params_b:  float        # total parameter count in billions
    label:     str          # e.g. "7B"
    context_k: int          # max supported context (K tokens)
    use_case:  str
    families:  list[str]    # representative real models at this size
    min_quant: str          # lowest quant still worth using
    arch:      ModelArch    # transformer dimensions (see ModelArch docstring)


#                                     layers  kv_heads  head_dim  hidden  vocab
MODEL_FAMILIES: list[ModelFamily] = [
    ModelFamily(0.5,  "0.5B",  4,  "Nano / edge device",
                ["Qwen2.5-0.5B", "SmolLM-360M"], "Q4_K_M",
                ModelArch(24,   2,  64,   896, 151936)),

    ModelFamily(1.0,  "1B",    8,  "Tiny assistant, code completion",
                ["Llama-3.2-1B", "Qwen2.5-1.5B"], "Q4_K_M",
                ModelArch(16,   8,  64,  2048, 128256)),

    ModelFamily(3.0,  "3B",    8,  "Light chat / on-device assistant",
                ["Llama-3.2-3B", "Phi-3.5-mini", "Qwen2.5-3B"], "Q4_K_M",
                ModelArch(28,   8, 128,  3072, 128256)),

    ModelFamily(7.0,  "7B",    8,  "General assistant (sweet-spot)",
                ["Mistral-7B-v0.3", "Llama-3.1-8B", "Qwen2.5-7B", "Gemma-2-9B"], "Q3_K_M",
                ModelArch(32,   8, 128,  4096, 128256)),

    ModelFamily(14.0, "14B",  16,  "Strong reasoning & coding",
                ["Qwen2.5-14B", "Phi-4-14B", "Gemma-2-12B"], "Q3_K_M",
                ModelArch(48,   8, 128,  5120, 151936)),

    ModelFamily(22.0, "22B",  16,  "Near-GPT-3.5 quality",
                ["Qwen2.5-22B", "Mistral-Small-22B"], "Q3_K_M",
                ModelArch(48,   8, 128,  6144,  32000)),

    ModelFamily(32.0, "32B",  32,  "Instruction-tuned powerhouse",
                ["Qwen2.5-32B", "Mistral-Small-3.1-32B"], "Q3_K_S",
                ModelArch(64,   8, 128,  5120, 151936)),

    ModelFamily(47.0, "47B",  32,  "MoE efficiency (Mixtral-8x7B tier)",
                ["Mixtral-8x7B"], "Q3_K_S",
                ModelArch(32,   8, 128,  4096,  32000)),

    ModelFamily(70.0, "70B",  64,  "Frontier open-source quality",
                ["Llama-3.1-70B", "Qwen2.5-72B", "Nemotron-70B"], "Q3_K_S",
                ModelArch(80,   8, 128,  8192, 128256)),

    ModelFamily(123.0,"123B", 128, "Very large; top open-weight quality",
                ["Mistral-Large-2", "Mistral-Large-Instruct-2407"], "Q2_K",
                ModelArch(88,   8, 128, 12288,  32768)),

    ModelFamily(236.0,"236B",  64, "Huge MoE (Mixtral-8x22B tier)",
                ["Mixtral-8x22B"], "IQ2_XS",
                ModelArch(56,   8, 128,  6144,  32000)),

    ModelFamily(405.0,"405B", 128, "Largest open-weight (Llama-3.1-405B)",
                ["Llama-3.1-405B"], "IQ1_S",
                ModelArch(126,  8, 128, 16384, 128256)),
]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HARDWARE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index:          int
    name:           str
    vram_gb:        float
    driver_version: str = ""
    cuda_version:   str = ""


def _parse_nvidia_smi() -> list[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        gpus: list[GPUInfo] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                vram_gb=round(float(parts[2]) / 1024, 2),
                driver_version=parts[3],
            ))
        try:
            banner = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL)
            m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
            if m:
                for g in gpus:
                    g.cuda_version = m.group(1)
        except Exception:
            pass
        return gpus
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def _parse_rocm() -> list[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            text=True, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        gpus: list[GPUInfo] = []
        for k, v in data.items():
            if "card" in k.lower():
                vram_bytes = int(v.get("VRAM Total Memory (B)", 0))
                gpus.append(GPUInfo(
                    index=len(gpus),
                    name=v.get("Card series", "AMD GPU"),
                    vram_gb=round(vram_bytes / (1024 ** 3), 2),
                ))
        return gpus
    except Exception:
        return []


def _parse_metal() -> list[GPUInfo]:
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            text=True, stderr=subprocess.DEVNULL
        )
        data = json.loads(out)
        hw = data.get("SPHardwareDataType", [{}])[0]
        chip = hw.get("chip_type", hw.get("cpu_type", "Apple Silicon"))
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        return [GPUInfo(index=0, name=chip, vram_gb=round(total_ram * 0.75, 1))]
    except Exception:
        # Fallback: at least report something for Apple Silicon
        if platform.processor() == "arm":
            total_ram = psutil.virtual_memory().total / (1024 ** 3)
            return [GPUInfo(index=0, name="Apple Silicon", vram_gb=round(total_ram * 0.75, 1))]
        return []


@dataclass
class SystemInfo:
    ram_gb:           float
    os:               str
    cpu:              str
    cpu_cores:        int
    cpu_freq_ghz:     float
    gpus:             list[GPUInfo]
    has_avx2:         bool
    has_avx512:       bool
    is_apple_silicon: bool

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        return max(self.gpus, key=lambda g: g.vram_gb) if self.gpus else None

    @property
    def gpu_names(self) -> str:
        return ", ".join(g.name for g in self.gpus) if self.gpus else "None"


def detect_system() -> SystemInfo:
    mem = psutil.virtual_memory()
    ram = round(mem.total / (1024 ** 3), 1)

    cpu_freq = psutil.cpu_freq()
    freq_ghz = round((cpu_freq.max or cpu_freq.current) / 1000, 2) if cpu_freq else 0.0

    is_apple = platform.processor() == "arm" and platform.system() == "Darwin"

    has_avx2 = has_avx512 = False
    try:
        if platform.system() == "Linux":
            flags = Path("/proc/cpuinfo").read_text()
            has_avx2   = " avx2 "   in flags
            has_avx512 = " avx512f " in flags
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-a", "machdep.cpu.leaf7_features"],
                text=True, stderr=subprocess.DEVNULL
            )
            has_avx2   = "AVX2"    in out
            has_avx512 = "AVX512F" in out
        elif platform.system() == "Windows":
            # cpuid via wmic — best effort
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Caption"], text=True, stderr=subprocess.DEVNULL
            )
            # can't reliably detect AVX2 on Windows without third-party tools
    except Exception:
        pass

    gpus: list[GPUInfo] = []
    if shutil.which("nvidia-smi"):
        gpus = _parse_nvidia_smi()
    elif shutil.which("rocm-smi"):
        gpus = _parse_rocm()
    elif is_apple:
        gpus = _parse_metal()

    return SystemInfo(
        ram_gb=ram,
        os=f"{platform.system()} {platform.release()}",
        cpu=platform.processor() or platform.machine(),
        cpu_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        cpu_freq_ghz=freq_ghz,
        gpus=gpus,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        is_apple_silicon=is_apple,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MEMORY BUDGET  (raw usable pool; per-model costs deducted separately)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryBudget:
    source:    str     # "GPU VRAM" | "Apple Unified Memory" | "System RAM (CPU-only)"
    total_gb:  float
    usable_gb: float   # after OS/driver efficiency factor only
    note:      str


def compute_budget(sys: SystemInfo) -> MemoryBudget:
    if sys.gpus and not sys.is_apple_silicon:
        total = sys.total_vram_gb
        return MemoryBudget(
            source="GPU VRAM",
            total_gb=round(total, 2),
            usable_gb=round(total * _GPU_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB,
            note=sys.gpu_names,
        )
    if sys.is_apple_silicon:
        total = sys.ram_gb
        return MemoryBudget(
            source="Apple Unified Memory",
            total_gb=round(total, 2),
            usable_gb=round(total * _APPLE_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB,
            note="Apple Silicon – CPU and GPU share the same physical RAM pool",
        )
    total = sys.ram_gb
    return MemoryBudget(
        source="System RAM (CPU-only)",
        total_gb=round(total, 2),
        usable_gb=round(total * _RAM_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB,
        note="No GPU detected – llama.cpp will run in CPU-only mode",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PER-MODEL MEMORY OVERHEAD
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MemoryOverhead:
    """
    All non-weight memory costs for a specific (model, context) combination.

    kv_cache_gb:
        Exact formula: 2 × layers × kv_heads × head_dim × ctx_tokens × dtype_bytes
        At fp16 (2 bytes), a 7B model at 4K context ≈ 0.5 GB; at 32K ≈ 4 GB.
        This scales linearly with context length – the single biggest variable.

    cpu_compute_gb:
        llama.cpp allocates a ggml compute graph buffer on the CPU.
        Dominant term: attention score matrix = num_heads × n_batch × n_ctx × 4 bytes.
        Zero for GPU/Apple (device memory; negligible vs VRAM).

    linux_cache_gb:
        On Linux, llama.cpp uses mmap by default. Model pages enter the kernel
        page cache and compete with other processes' working sets.
        Measured live from /proc/meminfo (buffers + cached) × 0.30.
        Zero on GPU (VRAM is separate) and Apple (unified; macOS manages this).

    tokenizer_gb:
        llama.cpp binary + sentencepiece runtime + vocab buffer.
        Flat 200 MB; negligible but real.
    """
    kv_cache_gb:    float
    kv_context_k:   int
    kv_dtype_bytes: int
    cpu_compute_gb: float
    linux_cache_gb: float
    tokenizer_gb:   float

    @property
    def total_gb(self) -> float:
        return round(
            self.kv_cache_gb + self.cpu_compute_gb +
            self.linux_cache_gb + self.tokenizer_gb, 3
        )

    def to_dict(self) -> dict:
        kv_dtype_label = "fp16" if self.kv_dtype_bytes == 2 else "int8"
        return {
            "kv_cache_gb":       round(self.kv_cache_gb,    3),
            "kv_context_k":      self.kv_context_k,
            "kv_dtype":          kv_dtype_label,
            "cpu_compute_gb":    round(self.cpu_compute_gb, 3),
            "linux_cache_gb":    round(self.linux_cache_gb, 3),
            "tokenizer_gb":      round(self.tokenizer_gb,   3),
            "total_overhead_gb": self.total_gb,
        }


def _measure_linux_cache_pressure_gb() -> float:
    """
    Lightweight estimate of *non-reclaimable* cache pressure.
    """
    try:
        mem = psutil.virtual_memory()

        cached_gb = (
            getattr(mem, "cached", 0) +
            getattr(mem, "buffers", 0)
        ) / (1024 ** 3)

        # Only treat a small fraction as "sticky"
        sticky = cached_gb * 0.10   # 10% instead of 30%

        return round(max(0.3, sticky), 2)

    except Exception:
        return 0.5


def compute_overhead(
    model: ModelFamily,
    context_k: int,
    kv_dtype_bytes: int,
    budget: MemoryBudget,
) -> MemoryOverhead:
    a = model.arch
    n_ctx = context_k * 1024

    is_gpu   = budget.source == "GPU VRAM"
    is_apple = budget.source == "Apple Unified Memory"
    is_linux = platform.system() == "Linux"

    # ── 1. KV cache ────────────────────────────────────────────────────────────
    # Per llama.cpp: two tensors (K, V) per layer, each (n_kv_heads, n_ctx, head_dim)
    kv_bytes = 2 * a.num_layers * a.num_kv_heads * a.head_dim * n_ctx * kv_dtype_bytes
    kv_gb    = kv_bytes / (1024 ** 3)

    # ── 2. CPU compute buffer ──────────────────────────────────────────────────
    # Dominant ggml term: attention score matrix Q·Kᵀ
    #   shape: (num_heads, n_batch, n_ctx) × float32
    # GPU/Apple: this lives in device memory at ~0 host-RAM cost.
    # CPU: must fit in system RAM; grows linearly with ctx_len and batch size.
    cpu_compute_gb = 0.0
    if not is_gpu and not is_apple:
        attn_bytes = a.num_heads * LLAMA_N_BATCH * n_ctx * 4
        # ggml also allocates Q,K,V projection buffers (~3 × hidden × n_batch × fp32)
        proj_bytes = 3 * a.hidden_size * LLAMA_N_BATCH * 4
        cpu_compute_gb = max(0.10, (attn_bytes + proj_bytes) / (1024 ** 3))
        cpu_compute_gb = round(cpu_compute_gb, 3)

    # ── 3. Linux page-cache pressure ───────────────────────────────────────────
    linux_cache_gb = 0.0
    if platform.system() == "Linux" and not is_gpu and not is_apple:
        linux_cache_gb = _measure_linux_cache_pressure_gb()

    return MemoryOverhead(
        kv_cache_gb=round(kv_gb,         3),
        kv_context_k=context_k,
        kv_dtype_bytes=kv_dtype_bytes,
        cpu_compute_gb=cpu_compute_gb,
        linux_cache_gb=linux_cache_gb,
        tokenizer_gb=TOKENIZER_OVERHEAD_GB,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  BANDWIDTH & TOKENS-PER-SECOND ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────
# Single-token generation is purely memory-bandwidth-bound: every weight is
# read exactly once per generated token.  Formula: BW_GB_s / (2 × model_GB).
# The ×2 accounts for the round-trip (read weight → compute → write activation)
# and typical cache-line waste for non-sequential access patterns.

_GPU_BW: list[tuple[str, float]] = [
    # ── NVIDIA Data-centre ──
    ("H200",        4800.0), ("H100 SXM",  3350.0), ("H100 PCIe",  2000.0),
    ("A100 SXM 80", 2000.0), ("A100 PCIe 80", 1935.0), ("A100",    1555.0),
    ("A40",          696.0), ("A30",         933.0),
    # ── RTX 40xx ──
    ("4090",        1008.0), ("4080 SUPER",  736.0), ("4080",        716.0),
    ("4070 Ti SUPER",672.0), ("4070 Ti",    504.0),
    ("4070 SUPER",   504.0), ("4070",        504.0),
    ("4060 Ti",      288.0), ("4060",        272.0),
    # ── RTX 40 Laptop ──
    ("4090 Laptop",  576.0), ("4080 Laptop", 432.0),
    ("4070 Laptop",  288.0), ("4060 Laptop", 192.0),
    # ── RTX 30xx ──
    ("3090 Ti",     1008.0), ("3090",        936.0),
    ("3080 Ti",      912.0), ("3080 12GB",   912.0), ("3080",       760.0),
    ("3070 Ti",      608.0), ("3070",        448.0),
    ("3060 Ti",      448.0), ("3060",        360.0),
    # ── RTX 30 Laptop ──
    ("3080 Laptop",  384.0), ("3070 Laptop", 256.0), ("3060 Laptop",192.0),
    # ── RTX 20xx ──
    ("2080 Ti",      616.0), ("2080 SUPER",  496.0), ("2080",       448.0),
    ("2070 SUPER",   448.0), ("2070",        448.0),
    ("2060 SUPER",   448.0), ("2060",        336.0),
    # ── GTX ──
    ("1080 Ti",      484.0), ("1080",        320.0),
    ("1070 Ti",      256.0), ("1070",        256.0), ("1060",       192.0),
    # ── Professional ──
    ("A6000",        768.0), ("A5000",       768.0),
    ("A4000",        448.0), ("A2000",       288.0),
    # ── AMD ──
    ("MI300X",      5300.0), ("MI250X",     3277.0), ("MI210",     1600.0),
    ("RX 7900 XTX",  960.0), ("RX 7900 XT",  800.0),
    ("RX 7800 XT",   624.0), ("RX 7700 XT",  432.0),
    ("RX 6900 XT",   512.0), ("RX 6800 XT",  512.0), ("RX 6700 XT",384.0),
]

_APPLE_BW: list[tuple[str, float]] = [
    ("M4 Ultra",  800.0), ("M4 Max",   300.0), ("M4 Pro",  120.0), ("M4",   75.0),
    ("M3 Ultra",  800.0), ("M3 Max",   300.0), ("M3 Pro",  150.0), ("M3",  100.0),
    ("M2 Ultra",  800.0), ("M2 Max",   400.0), ("M2 Pro",  200.0), ("M2",  100.0),
    ("M1 Ultra",  800.0), ("M1 Max",   400.0), ("M1 Pro",  200.0), ("M1",   68.0),
]


def _match_bw(name: str, table: list[tuple[str, float]], default: float) -> float:
    """Longest-match lookup: more specific entries in the table win."""
    name_up = name.upper()
    best_len, best_bw = 0, default
    for pattern, bw in table:
        if pattern.upper() in name_up and len(pattern) > best_len:
            best_len, best_bw = len(pattern), bw
    return best_bw


def _cpu_effective_bw_gb_s(sys_info: SystemInfo) -> float:
    """
    Estimate effective CPU memory bandwidth for llama.cpp inference.

    True bandwidth depends on DDR generation (undetectable without BIOS/SMBIOS),
    number of channels, and NUMA topology.  We use a conservative heuristic:

    Base bandwidth (dual-channel assumption):
      AVX-512 present  → likely server/HEDT with DDR4/DDR5 → 90 GB/s
      AVX2 + freq≥4.5  → likely DDR5 consumer               → 65 GB/s
      AVX2 + freq≥3.5  → likely DDR4                         → 50 GB/s
      AVX2             → older DDR4                           → 38 GB/s
      no AVX2          → very old system or scalar fallback   → 20 GB/s

    AVX2 penalty for llama.cpp:
      llama.cpp uses hand-written AVX2 SIMD kernels for quantised matmul.
      Without AVX2, it falls back to scalar which is ~2–3× slower, effectively
      halving the utilised bandwidth even if the DRAM bandwidth is adequate.
    """
    f = sys_info.cpu_freq_ghz

    if sys_info.has_avx512:
        bw = 90.0
    elif sys_info.has_avx2 and f >= 4.5:
        bw = 65.0
    elif sys_info.has_avx2 and f >= 3.5:
        bw = 50.0
    elif sys_info.has_avx2:
        bw = 38.0
    else:
        bw = 20.0   # scalar fallback

    # llama.cpp scalar-path penalty (no AVX2)
    if not sys_info.has_avx2:
        bw *= 0.50

    return bw


def estimate_tps(
    weights_gb: float,
    sys_info: SystemInfo,
    budget: MemoryBudget,
) -> float:
    """Rough tokens/sec estimate for single-token generation (decode phase)."""
    if weights_gb <= 0:
        return 0.0

    if budget.source == "GPU VRAM":
        gpu_name = sys_info.best_gpu.name if sys_info.best_gpu else ""
        bw = _match_bw(gpu_name, _GPU_BW, default=400.0)
    elif budget.source == "Apple Unified Memory":
        gpu_name = sys_info.gpus[0].name if sys_info.gpus else ""
        bw = _match_bw(gpu_name, _APPLE_BW, default=68.0)
    else:
        bw = _cpu_effective_bw_gb_s(sys_info)

    return round(bw / (2.0 * weights_gb), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  RECOMMENDATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Recommendation:
    quant:            Quant
    model:            ModelFamily
    weights_gb:       float
    overhead:         MemoryOverhead
    fits:             bool       # weights + overhead ≤ usable_gb
    fits_comfortably: bool       # headroom ≥ source-specific minimum (absolute GB)
    headroom_gb:      float      # usable_gb − (weights + overhead)
    speed_label:      str        # qualitative speed tier
    quality_stars:    int        # 1–5
    est_tps:          float      # estimated decode tokens/sec
    is_practical:     bool       # False if CPU && est_tps < MIN_PRACTICAL_TPS

    @property
    def total_gb(self) -> float:
        return round(self.weights_gb + self.overhead.total_gb, 2)


_TIER_STAR: dict[str, int] = {
    "ULTRA_FAST": 1, "FAST": 2, "BALANCED": 3, "QUALITY": 4, "REFERENCE": 5,
}

_SPEED_TABLE: list[tuple[float, str]] = [
    (2.5,  "Very Fast"),
    (3.5,  "Fast"),
    (5.0,  "Moderate"),
    (8.5,  "Somewhat Slow"),
    (99.0, "Slow"),
]


def _speed_label(bpw: float) -> str:
    for threshold, label in _SPEED_TABLE:
        if bpw <= threshold:
            return label
    return "Slow"


def build_recommendations(
    budget:        MemoryBudget,
    sys_info:      SystemInfo,
    context_k:     int,
    kv_dtype_bytes:int,
    min_bpw:       float = 0.0,
) -> list[Recommendation]:
    """
    For each quant type, find the largest ModelFamily whose
    weights + ALL overheads fit inside budget.usable_gb.
    """
    is_cpu    = budget.source == "System RAM (CPU-only)"
    comfort   = _COMFORT_HEADROOM_GB[budget.source]
    recs: list[Recommendation] = []

    for quant in QUANTS:
        if quant.bpw < min_bpw:
            continue

        # Walk from smallest to largest; keep the last one that fits
        best_model: Optional[ModelFamily] = None
        for mf in MODEL_FAMILIES:
            overhead = compute_overhead(mf, context_k, kv_dtype_bytes, budget)
            total    = mf.params_b * quant.factor + overhead.total_gb
            if total <= budget.usable_gb:
                best_model = mf

        if best_model is None:
            continue

        weights_gb = best_model.params_b * quant.factor
        overhead   = compute_overhead(best_model, context_k, kv_dtype_bytes, budget)
        total_gb   = weights_gb + overhead.total_gb
        headroom = budget.usable_gb - total_gb
        tps        = estimate_tps(weights_gb, sys_info, budget)

        recs.append(Recommendation(
            quant=quant,
            model=best_model,
            weights_gb=round(weights_gb, 2),
            overhead=overhead,
            fits=total_gb <= budget.usable_gb,
            fits_comfortably = headroom >= _COMFORT_HEADROOM_GB[budget.source],
            headroom_gb=round(headroom, 2),
            speed_label=_speed_label(quant.bpw),
            quality_stars=_TIER_STAR.get(quant.tier, 3),
            est_tps=tps,
            is_practical=(tps >= MIN_PRACTICAL_TPS) if is_cpu else True,
        ))

    return recs


def top_picks(recs: list[Recommendation]) -> dict[str, Recommendation]:
    """
    One best recommendation per tier.
    Priority: comfortable + practical > comfortable > practical > any fit.
    Within a priority bucket, prefer larger model then higher bpw.
    """
    picks: dict[str, Recommendation] = {}
    for tier in ("ULTRA_FAST", "FAST", "BALANCED", "QUALITY", "REFERENCE"):
        tier_recs = [r for r in recs if r.quant.tier == tier and r.fits]
        if not tier_recs:
            continue
        for subset in [
            [r for r in tier_recs if r.fits_comfortably and r.is_practical],
            [r for r in tier_recs if r.fits_comfortably],
            [r for r in tier_recs if r.is_practical],
            tier_recs,
        ]:
            if subset:
                picks[tier] = max(subset, key=lambda r: (r.model.params_b, r.quant.bpw))
                break
    return picks


# ──────────────────────────────────────────────────────────────────────────────
# 8.  JSON OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def build_json_output(
    sys_info: SystemInfo,
    budget:   MemoryBudget,
    recs:     list[Recommendation],
    picks:    dict[str, Recommendation],
    context_k:     int,
    kv_dtype_bytes:int,
) -> dict:
    def gpu_dict(g: GPUInfo) -> dict:
        d: dict = {"name": g.name, "vram_gb": g.vram_gb}
        if g.driver_version: d["driver"] = g.driver_version
        if g.cuda_version:   d["cuda"]   = g.cuda_version
        return d

    return {
        "system": {
            "os":                 sys_info.os,
            "cpu":                sys_info.cpu,
            "cpu_cores_physical": sys_info.cpu_cores,
            "cpu_freq_ghz":       sys_info.cpu_freq_ghz,
            "ram_gb":             sys_info.ram_gb,
            "avx2":               sys_info.has_avx2,
            "avx512":             sys_info.has_avx512,
            "apple_silicon":      sys_info.is_apple_silicon,
            "gpus":               [gpu_dict(g) for g in sys_info.gpus],
        },
        "memory_budget": {
            "source":     budget.source,
            "total_gb":   budget.total_gb,
            "usable_gb":  budget.usable_gb,
            "note":       budget.note,
        },
        "inference_settings": {
            "context_k":       context_k,
            "kv_dtype":        "fp16" if kv_dtype_bytes == 2 else "int8",
            "kv_dtype_bytes":  kv_dtype_bytes,
        },
        "top_picks": {
            tier: {
                "quant":            r.quant.name,
                "bpw":              r.quant.bpw,
                "model":            r.model.label,
                "params_b":         r.model.params_b,
                "weights_gb":       r.weights_gb,
                "overhead":         r.overhead.to_dict(),
                "total_memory_gb":  r.total_gb,
                "headroom_gb":      r.headroom_gb,
                "fits_comfortably": r.fits_comfortably,
                "est_tps":          r.est_tps,
                "is_practical":     r.is_practical,
                "speed":            r.speed_label,
                "quality_stars":    r.quality_stars,
                "example_models":   r.model.families[:3],
                "use_case":         r.model.use_case,
                "quality_note":     r.quant.quality_note,
            }
            for tier, r in picks.items()
        },
        "full_quant_table": [
            {
                "quant":            r.quant.name,
                "tier":             r.quant.tier,
                "bpw":              r.quant.bpw,
                "best_model":       r.model.label,
                "weights_gb":       r.weights_gb,
                "kv_cache_gb":      r.overhead.kv_cache_gb,
                "cpu_compute_gb":   r.overhead.cpu_compute_gb,
                "linux_cache_gb":   r.overhead.linux_cache_gb,
                "total_memory_gb":  r.total_gb,
                "fits_comfortably": r.fits_comfortably,
                "headroom_gb":      r.headroom_gb,
                "est_tps":          r.est_tps,
                "is_practical":     r.is_practical,
                "speed":            r.speed_label,
                "quality_stars":    r.quality_stars,
            }
            for r in recs
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9.  MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────────────────────

_STARS = {1: "★☆☆☆☆", 2: "★★☆☆☆", 3: "★★★☆☆", 4: "★★★★☆", 5: "★★★★★"}
_TIER_EMOJI = {
    "ULTRA_FAST": "⚡", "FAST": "🚀", "BALANCED": "⚖️",
    "QUALITY": "💎", "REFERENCE": "🔬",
}


def generate_markdown(
    sys_info:      SystemInfo,
    budget:        MemoryBudget,
    recs:          list[Recommendation],
    picks:         dict[str, Recommendation],
    context_k:     int,
    kv_dtype_bytes:int,
) -> str:
    lines: list[str] = []
    a = lines.append
    is_cpu = budget.source == "System RAM (CPU-only)"

    a("# 🤖 LLM Hardware Compatibility Report")
    a("")
    a("> Auto-generated by `model_recommender.py` — llama.cpp-aware memory model")
    a("")

    # ── System ─────────────────────────────────────────────────────────────────
    a("## 🖥️ System")
    a("")
    a("| Property | Value |")
    a("|----------|-------|")
    a(f"| OS | `{sys_info.os}` |")
    a(f"| CPU | `{sys_info.cpu}` · {sys_info.cpu_cores} physical cores @ {sys_info.cpu_freq_ghz} GHz |")
    a(f"| RAM | **{sys_info.ram_gb} GB** |")
    a(f"| AVX2 / AVX-512 | {sys_info.has_avx2} / {sys_info.has_avx512} |")
    if sys_info.gpus:
        for g in sys_info.gpus:
            drv = f" · driver {g.driver_version}" if g.driver_version else ""
            cuda = f", CUDA {g.cuda_version}" if g.cuda_version else ""
            a(f"| GPU {g.index} | **{g.name}** · {g.vram_gb} GB VRAM{drv}{cuda} |")
    else:
        a("| GPU | None detected (CPU-only mode) |")
    a("")

    if not sys_info.has_avx2 and is_cpu:
        a("> ⚠️ **No AVX2 detected.** llama.cpp will use scalar fallback kernels, "
          "which are ~2–3× slower than AVX2. Performance estimates are adjusted accordingly.")
        a("")

    # ── Memory budget ──────────────────────────────────────────────────────────
    kv_dtype_label = "fp16" if kv_dtype_bytes == 2 else "int8 (q8_0)"
    a("## 💾 Memory Budget")
    a("")
    a("| Item | Value |")
    a("|------|-------|")
    a(f"| Source | {budget.source} |")
    a(f"| Physical total | {budget.total_gb} GB |")
    a(f"| After OS/driver overhead | **{budget.usable_gb} GB** |")
    a(f"| Target context | {context_k}K tokens |")
    a(f"| KV-cache dtype | {kv_dtype_label} ({kv_dtype_bytes} bytes/elem) |")
    a(f"| Note | {budget.note} |")
    a("")
    a("> Per-model costs (KV cache, compute buffer, tokenizer) are deducted per recommendation below.")
    a("")

    # ── Top picks ──────────────────────────────────────────────────────────────
    a("## 🏆 Top Picks (one per tier)")
    a("")
    for tier, r in picks.items():
        emoji = _TIER_EMOJI.get(tier, "•")
        warn = " ⚠️ *impractical on CPU*" if not r.is_practical else ""
        a(f"### {emoji} {tier.replace('_', ' ').title()}{warn}")
        a(f"| Field | Value |")
        a(f"|-------|-------|")
        a(f"| Quant | `{r.quant.name}` ({r.quant.bpw:.2f} bpw) |")
        a(f"| Model | **{r.model.label}** ({r.model.params_b}B params) |")
        a(f"| Weights | {r.weights_gb} GB |")
        a(f"| KV cache @ {r.overhead.kv_context_k}K | {r.overhead.kv_cache_gb:.3f} GB |")
        if r.overhead.cpu_compute_gb > 0:
            a(f"| CPU compute buffer | {r.overhead.cpu_compute_gb:.3f} GB |")
        if r.overhead.linux_cache_gb > 0:
            a(f"| Linux page-cache reserve | {r.overhead.linux_cache_gb:.3f} GB |")
        a(f"| Tokenizer overhead | {r.overhead.tokenizer_gb:.2f} GB |")
        a(f"| **Total memory** | **{r.total_gb} GB** |")
        a(f"| Headroom | {r.headroom_gb:.2f} GB {'✅' if r.fits_comfortably else '⚠️'} |")
        a(f"| Est. decode speed | ~{r.est_tps} t/s |")
        a(f"| Quality | {_STARS.get(r.quality_stars, '')} |")
        a(f"| Use case | {r.model.use_case} |")
        a(f"| Example models | {', '.join(r.model.families[:3])} |")
        a(f"| Notes | {r.quant.quality_note} |")
        a("")

    # ── Full table ─────────────────────────────────────────────────────────────
    a("## 📊 Full Quant × Model Table")
    a("")
    a("| Quant | Tier | BPW | Model | Weights | KV$ | CPU buf | Total | Comfortable | ~t/s | Quality |")
    a("|-------|------|-----|-------|---------|-----|---------|-------|:-----------:|------|---------|")
    for r in recs:
        comfort = "✅" if r.fits_comfortably else "⚠️"
        practical = "" if r.is_practical else " 🐢"
        a(f"| `{r.quant.name}` | {r.quant.tier} | {r.quant.bpw:.2f} | {r.model.label} "
          f"| {r.weights_gb:.1f} | {r.overhead.kv_cache_gb:.2f} "
          f"| {r.overhead.cpu_compute_gb:.2f} | {r.total_gb:.1f} "
          f"| {comfort} | {r.est_tps}{practical} | {_STARS.get(r.quality_stars,'')} |")
    a("")
    a("*KV$* = KV-cache GB at target context. "
      "*CPU buf* = ggml compute buffer (0 on GPU/Apple). "
      "*🐢* = estimated <1.5 t/s on CPU.")
    a("")

    # ── Device-specific tips ───────────────────────────────────────────────────
    a("## 💡 llama.cpp Tips")
    a("")
    common_tips = [
        "`Q4_K_M` is the community sweet-spot: great quality, reasonable size.",
        "`IQ4_XS` is slightly smaller than `Q4_K_M` with minimal quality loss (requires iMatrix).",
        "`Q8_0` is the highest-quality quant worth running; `F16` is overkill for inference.",
        "Reduce `--ctx-size` to shrink the KV cache linearly — halving context halves KV RAM.",
        "Use `--kv-cache-type q8_0` to halve KV cache size with ~0.5% quality loss.",
    ]
    if is_cpu:
        device_tips = [
            f"Set `--threads {sys_info.cpu_cores}` (physical cores only; hyperthreads hurt throughput).",
            "Add `--flash-attn` to dramatically reduce the ggml compute buffer at long contexts.",
            "Add `--no-mmap` on Linux if you experience stuttering from page-cache eviction.",
            "Prompt processing (prefill) benefits from `--n-batch 512`; generation speed does not.",
        ]
        if not sys_info.has_avx2:
            device_tips.insert(0,
                "⚠️ Compile llama.cpp with `-DLLAMA_AVX2=ON` if your CPU actually supports it — "
                "the packaged binary may not have it enabled."
            )
    elif budget.source == "Apple Unified Memory":
        device_tips = [
            "Use `--n-gpu-layers 999` to offload all layers to Metal for maximum speed.",
            "Add `--flash-attn` — Metal flash-attention is well-optimised on M-series.",
            "Avoid `--no-mmap`; mmap is efficient on macOS due to unified memory.",
        ]
    else:  # GPU
        device_tips = [
            "Use `--n-gpu-layers 999` to fully offload the model to VRAM.",
            "Add `--flash-attn` to reduce attention memory and allow longer contexts.",
            "With multiple GPUs, `--tensor-split` splits layers; single large GPU is usually faster.",
            "Compile llama.cpp with `cmake -DGGML_CUDA=ON` for best performance.",
        ]

    for tip in common_tips + device_tips:
        a(f"- {tip}")
    a("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 10.  CLI + MAIN
# ──────────────────────────────────────────────────────────────────────────────

def ask_yes_no(prompt: str) -> bool:
    try:
        return input(prompt + " [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect hardware and recommend GGUF quant + model for llama.cpp.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json-out",       default="quant_model_map.json")
    parser.add_argument("--md-out",         default="quant_model_report.md")
    parser.add_argument("--min-bpw",        type=float, default=0.0,
                        help="Exclude quants below this BPW (e.g. 3.0 skips 1–2 bit quants)")
    parser.add_argument("--context",        type=int, default=DEFAULT_CONTEXT_K,
                        help="Target inference context in K tokens (e.g. 4 = 4096 tokens)")
    parser.add_argument("--kv-bits",        type=int, default=DEFAULT_KV_BITS, choices=[8, 16],
                        help="KV-cache dtype: 16 = fp16 (default), 8 = int8 (--kv-cache-type q8_0)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive prompts; suppress Markdown output")
    parser.add_argument("--write-md",       action="store_true",
                        help="Always write Markdown report without prompting")
    args = parser.parse_args()

    kv_dtype_bytes = args.kv_bits // 8   # 16-bit → 2 bytes, 8-bit → 1 byte

    print("🔍 Detecting hardware…", flush=True)
    sys_info = detect_system()
    budget   = compute_budget(sys_info)

    print(f"   {budget.source}: {budget.total_gb} GB total → {budget.usable_gb} GB usable")
    if sys_info.gpus:
        print(f"   GPU(s): {sys_info.gpu_names}")
    print(f"   Context: {args.context}K tokens · KV dtype: fp{args.kv_bits}")
    print("", flush=True)

    recs  = build_recommendations(budget, sys_info, args.context, kv_dtype_bytes, args.min_bpw)
    picks = top_picks(recs)

    # ── JSON ───────────────────────────────────────────────────────────────────
    output    = build_json_output(sys_info, budget, recs, picks, args.context, kv_dtype_bytes)
    json_path = Path(args.json_out)
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\n✅  JSON  → {json_path}", flush=True)

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"  {'QUICK SUMMARY':^66}")
    print("─" * 70)
    print(f"  Memory source : {budget.source}")
    print(f"  Usable budget : {budget.usable_gb} GB")
    print(f"  Context       : {args.context}K tokens · KV fp{args.kv_bits}")
    print("─" * 70)
    hdr = f"  {'Tier':<14}  {'Quant':<12}  {'Model':<7}  {'Total GB':>8}  {'Headroom':>9}  {'~t/s':>6}"
    print(hdr)
    print("  " + "─" * 66)
    for tier, r in picks.items():
        practical_flag = "" if r.is_practical else " 🐢"
        comfort_flag   = "✅" if r.fits_comfortably else "⚠️ "
        print(
            f"  {tier:<14}  {r.quant.name:<12}  {r.model.label:<7}"
            f"  {r.total_gb:>7.1f} GB"
            f"  {r.headroom_gb:>6.1f} GB {comfort_flag}"
            f"  {r.est_tps:>5.0f}{practical_flag}"
        )
    print("─" * 70)

    # ── Markdown ───────────────────────────────────────────────────────────────
    write_md = args.write_md
    if not write_md and not args.no_interactive:
        write_md = ask_yes_no("\n📄 Save a human-readable Markdown report?")

    if write_md:
        md_path = Path(args.md_out)
        md_path.write_text(
            generate_markdown(sys_info, budget, recs, picks, args.context, kv_dtype_bytes),
            encoding="utf-8",
        )
        print(f"✅  Markdown → {md_path}")


if __name__ == "__main__":
    main()