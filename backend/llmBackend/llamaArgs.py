from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import psutil
except ImportError:
    raise ImportError(
        "psutil is required for llama_param_calculator:  pip install psutil"
    ) from None

__all__ = [
    "QUANT_BPW",
    "GPUInfo",
    "HostInfo",
    "LlamaParams",
    "RecommendationResult",
    "detect_host",
    "calculate_params",
    "recommend",
    "generate_markdown",
]


# ──────────────────────────────────────────────────────────────────────────────
# QUANT TABLE  (bits-per-weight; needed to convert params ↔ size and for KV-bit hints)
# ──────────────────────────────────────────────────────────────────────────────

QUANT_BPW: dict[str, float] = {
    "IQ1_S": 1.56, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_M": 2.57, "Q2_K": 2.63,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.30, "Q3_K_S": 3.50, "Q3_K_M": 3.91, "Q3_K_L": 4.27,
    "IQ4_XS": 4.25, "IQ4_NL": 4.50, "Q4_0": 4.55, "Q4_K_S": 4.37, "Q4_K_M": 4.85, "Q4_K_L": 4.90,
    "Q5_K_S": 5.54, "Q5_K_M": 5.68, "Q6_K": 6.57, "Q8_0": 8.50, "F16": 16.00,
}

GGUF_OVERHEAD_FRACTION = 0.04   # metadata, tokenizer, tensors padding ≈ 3-5% on top of raw weights



# ──────────────────────────────────────────────────────────────────────────────
# HARDWARE DETECTION  (trimmed: only what parameter selection needs)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    name: str
    vram_gb: float


@dataclass
class HostInfo:
    ram_gb: float
    available_ram_gb: float
    physical_cores: int
    logical_cores: int
    has_avx2: bool
    is_apple_silicon: bool
    gpus: list[GPUInfo]

    @property
    def total_vram_gb(self) -> float:
        return sum(g.vram_gb for g in self.gpus)

    @property
    def gpu_names(self) -> str:
        return ", ".join(g.name for g in self.gpus) if self.gpus else "None"


def _nvidia_gpus() -> list[GPUInfo]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        gpus = []
        for line in out.strip().splitlines():
            name, vram = [p.strip() for p in line.split(",")]
            gpus.append(GPUInfo(name=name, vram_gb=round(float(vram) / 1024, 2)))
        return gpus
    except Exception:
        return []


def _apple_gpu(ram_gb: float) -> list[GPUInfo]:
    if platform.system() != "Darwin" or platform.processor() != "arm":
        return []
    chip = "Apple Silicon"
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "-json"],
            text=True, stderr=subprocess.DEVNULL,
        )
        hw = json.loads(out).get("SPHardwareDataType", [{}])[0]
        chip = hw.get("chip_type", chip)
    except Exception:
        pass
    # Unified memory: ~75% of total RAM is realistically usable for GPU workloads
    return [GPUInfo(name=chip, vram_gb=round(ram_gb * 0.75, 1))]


def detect_host(sample_seconds: float = 0.0) -> HostInfo:
    mem = psutil.virtual_memory()
    ram_gb = round(mem.total / (1024 ** 3), 2)

    if sample_seconds > 0:
        import time
        readings = []
        n = max(2, int(sample_seconds / 0.5))
        for i in range(n):
            readings.append(psutil.virtual_memory().available / (1024 ** 3))
            if i < n - 1:
                time.sleep(0.5)
        available_gb = round(sum(readings) / len(readings), 2)
    else:
        available_gb = round(mem.available / (1024 ** 3), 2)

    has_avx2 = False
    try:
        if platform.system() == "Linux":
            has_avx2 = " avx2 " in Path("/proc/cpuinfo").read_text()
        elif platform.system() == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-a", "machdep.cpu.leaf7_features"],
                text=True, stderr=subprocess.DEVNULL,
            )
            has_avx2 = "AVX2" in out
    except Exception:
        pass

    is_apple = platform.processor() == "arm" and platform.system() == "Darwin"

    gpus: list[GPUInfo] = []
    if shutil.which("nvidia-smi"):
        gpus = _nvidia_gpus()
    elif is_apple:
        gpus = _apple_gpu(ram_gb)

    return HostInfo(
        ram_gb=ram_gb,
        available_ram_gb=available_gb,
        physical_cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        has_avx2=has_avx2,
        is_apple_silicon=is_apple,
        gpus=gpus,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LlamaParams:
    model_path:      str
    n_threads:       int
    n_threads_batch: int
    n_ctx:           int
    n_batch:         int
    n_gpu_layers:    int
    use_mmap:        bool
    use_mlock:       bool
    verbose:         bool
    # explanatory metadata, not passed to Llama() itself
    reasoning: dict[str, str]

    def to_constructor_kwargs(self) -> dict:
        return {
            "model_path": self.model_path,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads_batch,
            "n_ctx": self.n_ctx,
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": self.verbose,
        }


def _estimate_total_layers(params_b: float) -> int:
    """
    Rough layer-count heuristic from parameter count, used only to translate
    a partial GPU offload budget into a sensible n_gpu_layers integer when
    the caller doesn't supply --layers explicitly.

    Anchored to known real models (not a smooth formula, because real
    layer counts step in architecture-specific increments):
      0.5B≈24  1B≈16  3B≈28  7B≈32  14B≈48  22B≈48  32B≈64  70B≈80  123B≈88  405B≈126
    """
    anchors = [
        (0.5, 24), (1.0, 16), (3.0, 28), (7.0, 32), (14.0, 48),
        (22.0, 48), (32.0, 64), (47.0, 32), (70.0, 80),
        (123.0, 88), (236.0, 56), (405.0, 126),
    ]
    # nearest anchor by params_b
    closest = min(anchors, key=lambda a: abs(a[0] - params_b))
    return closest[1]


def calculate_params(
    model_path:    str,
    weight_gb:     float,
    quant:         str,
    host:          HostInfo,
    context:       int,
    params_b:      Optional[float] = None,
    total_layers:  Optional[int]   = None,
    verbose_flag:  bool = False,
) -> LlamaParams:
    reasoning: dict[str, str] = {}

    # ── infer params_b from weight size if not given ──
    bpw = QUANT_BPW.get(quant.upper())
    if bpw is None:
        raise ValueError(
            f"Unknown quant '{quant}'. Known: {', '.join(sorted(QUANT_BPW))}"
        )
    if params_b is None:
        # weight_gb ≈ params_b * bpw/8 * (1 + overhead)
        params_b = weight_gb / ((bpw / 8.0) * (1 + GGUF_OVERHEAD_FRACTION))
        reasoning["params_b"] = (
            f"Inferred {params_b:.2f}B params from {weight_gb} GB at {quant} "
            f"({bpw} bpw, +{GGUF_OVERHEAD_FRACTION*100:.0f}% GGUF overhead)."
        )
    else:
        reasoning["params_b"] = f"Given explicitly: {params_b}B params."

    layers = total_layers or _estimate_total_layers(params_b)
    if total_layers:
        reasoning["layers"] = f"Given explicitly: {layers} layers."
    else:
        reasoning["layers"] = (
            f"Estimated {layers} layers from {params_b:.1f}B params via nearest-known-model anchor "
            f"(no --layers supplied; pass it for exact results)."
        )

    # ── n_threads / n_threads_batch ──
    # Physical cores only: llama.cpp's AVX2 kernels saturate memory bandwidth well
    # before hyperthreads add anything; SMT siblings contend for the same cache lines
    # and execution ports, often *reducing* throughput at high core counts.
    n_threads = max(1, host.physical_cores)
    reasoning["n_threads"] = (
        f"= physical core count ({host.physical_cores}). Logical core count is "
        f"{host.logical_cores}; hyperthreads are excluded because llama.cpp's matmul "
        f"kernels are bandwidth-bound, not execution-port-bound — SMT siblings mostly "
        f"contend rather than add throughput."
    )
    n_threads_batch = n_threads
    reasoning["n_threads_batch"] = "= n_threads (prompt-processing scales the same way as generation)."

    # ── n_ctx ──
    n_ctx = context
    reasoning["n_ctx"] = f"Using requested context: {context} tokens."

    # ── GPU offload decision ──
    has_gpu = bool(host.gpus)
    vram_gb = host.total_vram_gb

    # Rough per-layer weight size (uniform split assumption)
    per_layer_gb = weight_gb / max(1, layers)

    if has_gpu:
        # Reserve headroom for KV cache + ggml compute buffer + driver context
        gpu_efficiency = 0.85 if not host.is_apple_silicon else 0.75
        usable_vram = vram_gb * gpu_efficiency

        # crude KV-cache estimate: ~ (n_ctx/1024) * 0.05 GB per layer at fp16
        # (matches typical 7B-class GQA models; conservative for larger hidden sizes)
        est_kv_gb = layers * (n_ctx / 1024) * 0.05
        compute_buf_gb = 0.3  # ggml graph buffer, GPU-resident, roughly model-size-independent

        budget_for_weights = max(0.0, usable_vram - est_kv_gb - compute_buf_gb)
        fittable_layers = int(budget_for_weights / per_layer_gb) if per_layer_gb > 0 else 0
        fittable_layers = max(0, min(layers, fittable_layers))

        if fittable_layers >= layers:
            n_gpu_layers = -1   # llama.cpp convention: -1 = offload all layers
            reasoning["n_gpu_layers"] = (
                f"-1 (all {layers} layers). Estimated VRAM need ≈ {weight_gb + est_kv_gb + compute_buf_gb:.2f} GB "
                f"(weights {weight_gb:.2f} + KV-cache ≈{est_kv_gb:.2f} + compute buffer ≈{compute_buf_gb:.2f}) "
                f"fits in {usable_vram:.2f} GB usable VRAM ({vram_gb:.2f} GB physical × {gpu_efficiency:.0%} "
                f"driver/headroom factor) on {host.gpu_names}."
            )
        elif fittable_layers > 0:
            n_gpu_layers = fittable_layers
            reasoning["n_gpu_layers"] = (
                f"{fittable_layers} of {layers} layers (partial offload). Full model + KV-cache + compute "
                f"buffer needs ≈{weight_gb + est_kv_gb + compute_buf_gb:.2f} GB but only {usable_vram:.2f} GB "
                f"usable VRAM is available on {host.gpu_names} ({vram_gb:.2f} GB physical). "
                f"Remaining {layers - fittable_layers} layers run on CPU; expect a real slowdown from "
                f"the CPU↔GPU handoff at the split boundary, not just blended bandwidth."
            )
        else:
            n_gpu_layers = 0
            reasoning["n_gpu_layers"] = (
                f"0 — even a single layer (≈{per_layer_gb:.3f} GB) plus KV-cache/compute overhead doesn't "
                f"comfortably fit in {usable_vram:.2f} GB usable VRAM on {host.gpu_names}. Running CPU-only."
            )
    else:
        n_gpu_layers = 0
        reasoning["n_gpu_layers"] = "0 — no GPU detected; CPU-only inference."

    # ── n_batch ──
    # Batch size mainly affects PROMPT PROCESSING (prefill) throughput, not
    # per-token generation speed. Bigger batches need a bigger ggml compute
    # buffer (roughly batch × hidden_size-ish × n_ctx-ish terms), so they're
    # capped by *available* (not total) RAM, not VRAM — the compute buffer
    # for partially/fully GPU-offloaded models still has a CPU-side staging
    # component in llama.cpp.
    available_for_batch_gb = max(0.5, host.available_ram_gb - weight_gb * (0 if n_gpu_layers == -1 else 1))
    if available_for_batch_gb >= 4:
        n_batch = 1024
    elif available_for_batch_gb >= 2:
        n_batch = 512
    elif available_for_batch_gb >= 1:
        n_batch = 256
    else:
        n_batch = 128
    reasoning["n_batch"] = (
        f"{n_batch} — chosen from ≈{available_for_batch_gb:.2f} GB RAM headroom left after "
        f"{'weights remain on GPU' if n_gpu_layers == -1 else 'CPU-resident weights'} "
        f"({host.available_ram_gb:.2f} GB available RAM measured). Larger batches speed up prompt "
        f"processing but raise compute-buffer RAM use roughly linearly; this caps it before swapping risk."
    )

    # ── use_mmap ──
    use_mmap = True
    reasoning["use_mmap"] = (
        "True — lets the OS lazily page in the GGUF file and share pages across processes; "
        "near-instant load time and lower peak RSS. Almost always the right default."
    )

    # ── use_mlock ──
    # Only worth it if the WHOLE working set (weights + KV-cache) comfortably
    # fits in *physical* RAM with real headroom left over — mlock pins pages
    # and prevents them from ever being swapped, which is good for latency
    # consistency but bad if it starves the rest of the system.
    est_kv_gb_for_mlock = (layers * (n_ctx / 1024) * 0.05) if not has_gpu else 0.0
    total_working_set = weight_gb + est_kv_gb_for_mlock
    mlock_margin_gb = host.ram_gb * 0.20  # keep 20% of total RAM free even when locking
    can_mlock = (
        n_gpu_layers != -1            # pointless to lock RAM pages for a fully GPU-resident model
        and host.ram_gb - total_working_set >= mlock_margin_gb
    )
    use_mlock = can_mlock
    if n_gpu_layers == -1:
        reasoning["use_mlock"] = "False — model is fully GPU-resident; locking host RAM pages has no benefit."
    elif can_mlock:
        reasoning["use_mlock"] = (
            f"True — estimated working set ≈{total_working_set:.2f} GB leaves "
            f"{host.ram_gb - total_working_set:.2f} GB free out of {host.ram_gb:.2f} GB total RAM, "
            f"above the {mlock_margin_gb:.2f} GB safety margin. Locking prevents the model from being "
            f"swapped out under memory pressure, trading flexibility for consistent latency."
        )
    else:
        reasoning["use_mlock"] = (
            f"False — estimated working set ≈{total_working_set:.2f} GB would leave less than "
            f"{mlock_margin_gb:.2f} GB free out of {host.ram_gb:.2f} GB total RAM if locked. "
            f"Locking here risks starving the OS and other processes; let pages be swappable instead."
        )

    return LlamaParams(
        model_path=model_path,
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        verbose=verbose_flag,
        reasoning=reasoning,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API  (the easiest entry point when importing this module)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    """Everything calculate_params() produces, bundled with the host it used."""
    host:      HostInfo
    params:    LlamaParams
    weight_gb: float
    quant:     str

    def to_constructor_kwargs(self) -> dict:
        """Shortcut: result.to_constructor_kwargs() instead of result.params.to_constructor_kwargs()."""
        return self.params.to_constructor_kwargs()

    def to_dict(self) -> dict:
        return {
            "host": asdict(self.host),
            "input": {"weight_gb": round(self.weight_gb, 3), "quant": self.quant},
            "constructor_kwargs": self.params.to_constructor_kwargs(),
            "reasoning": self.params.reasoning,
        }

    def to_markdown(self) -> str:
        return generate_markdown(self.host, self.params, self.weight_gb, self.quant)


def recommend(
    *,
    weight_gb:       Optional[float] = None,
    params_b:        Optional[float] = None,
    quant:           str,
    model_path:      str = "model.gguf",
    context:         int = 4096,
    total_layers:    Optional[int] = None,
    host:            Optional[HostInfo] = None,
    sample_seconds:  float = 0,
    verbose_llama:   bool = False,
) -> RecommendationResult:
    """
    Single-call convenience wrapper around detect_host() + calculate_params().

    This is the recommended entry point when using the module programmatically:

        from llama_param_calculator import recommend
        from llama_cpp import Llama

        result = recommend(weight_gb=4.37, quant="Q4_K_M", context=8192)
        llm = Llama(**result.to_constructor_kwargs())

    Args:
        weight_gb:      GGUF file size in GB. Mutually exclusive with params_b.
        params_b:       Parameter count in billions (weight_gb derived from quant). Mutually exclusive with weight_gb.
        quant:          Quant name, e.g. "Q4_K_M". Case-insensitive. Must be a key in QUANT_BPW.
        model_path:     Value to embed as model_path in the returned kwargs.
        context:         Desired n_ctx.
        total_layers:   Exact layer count if known; improves n_gpu_layers accuracy.
        host:           Reuse an already-detected HostInfo (e.g. across multiple recommend() calls)
                        instead of re-probing hardware every time. If omitted, detect_host() is called.
        sample_seconds: Only used when host is None. If > 0, averages available RAM over this many
                        seconds (clamped 10-30) instead of taking one instantaneous snapshot.
        verbose_llama:  Sets the `verbose` field in the returned Llama() kwargs.

    Returns:
        RecommendationResult with .params (LlamaParams), .host (HostInfo), and convenience
        .to_constructor_kwargs() / .to_dict() / .to_markdown() methods.

    Raises:
        ValueError: if quant is unknown, or if both/neither of weight_gb and params_b are given.
    """
    if (weight_gb is None) == (params_b is None):
        raise ValueError("Pass exactly one of weight_gb or params_b, not both/neither.")

    quant = quant.upper()
    if quant not in QUANT_BPW:
        raise ValueError(f"Unknown quant '{quant}'. Known: {', '.join(sorted(QUANT_BPW))}")

    if host is None:
        host = detect_host(sample_seconds=sample_seconds)

    if weight_gb is None:
        bpw = QUANT_BPW[quant]
        weight_gb = params_b * (bpw / 8.0) * (1 + GGUF_OVERHEAD_FRACTION)

    lp = calculate_params(
        model_path=model_path,
        weight_gb=weight_gb,
        quant=quant,
        host=host,
        context=context,
        params_b=params_b,
        total_layers=total_layers,
        verbose_flag=verbose_llama,
    )

    return RecommendationResult(host=host, params=lp, weight_gb=weight_gb, quant=quant)


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def print_report(host: HostInfo, lp: LlamaParams, weight_gb: float, quant: str) -> None:
    print("\n" + "─" * 70)
    print(f"  {'HOST':^66}")
    print("─" * 70)
    print(f"  RAM           : {host.ram_gb} GB total, {host.available_ram_gb} GB available")
    print(f"  CPU cores     : {host.physical_cores} physical / {host.logical_cores} logical")
    print(f"  AVX2          : {host.has_avx2}")
    print(f"  GPU(s)        : {host.gpu_names}"
          + (f"  ({host.total_vram_gb:.1f} GB VRAM total)" if host.gpus else ""))

    print("\n" + "─" * 70)
    print(f"  {'RECOMMENDED Llama() PARAMETERS':^66}")
    print("─" * 70)
    code = (
        "llm = Llama(\n"
        f"    model_path={lp.model_path!r},\n"
        f"    n_threads={lp.n_threads},\n"
        f"    n_threads_batch={lp.n_threads_batch},\n"
        f"    n_ctx={lp.n_ctx},\n"
        f"    n_batch={lp.n_batch},\n"
        f"    n_gpu_layers={lp.n_gpu_layers},\n"
        f"    use_mmap={lp.use_mmap},\n"
        f"    use_mlock={lp.use_mlock},\n"
        f"    verbose={lp.verbose},\n"
        ")"
    )
    print(code)

    print("\n" + "─" * 70)
    print(f"  {'REASONING':^66}")
    print("─" * 70)
    for key in ["params_b", "layers", "n_threads", "n_threads_batch",
                "n_ctx", "n_gpu_layers", "n_batch", "use_mmap", "use_mlock"]:
        if key in lp.reasoning:
            print(f"\n  • {key}:")
            print(f"      {lp.reasoning[key]}")
    print()


def generate_markdown(host: HostInfo, lp: LlamaParams, weight_gb: float, quant: str) -> str:
    lines: list[str] = []
    a = lines.append

    a("# 🦙 llama.cpp Parameter Recommendation")
    a("")
    a(f"> Calculated for a **{weight_gb} GB** model at **{quant}** quantization.")
    a("")

    a("## 🖥️ Host")
    a("")
    a("| Property | Value |")
    a("|----------|-------|")
    a(f"| RAM | {host.ram_gb} GB total, {host.available_ram_gb} GB available |")
    a(f"| CPU cores | {host.physical_cores} physical / {host.logical_cores} logical |")
    a(f"| AVX2 | {host.has_avx2} |")
    if host.gpus:
        for g in host.gpus:
            a(f"| GPU | {g.name} — {g.vram_gb} GB VRAM |")
    else:
        a("| GPU | None detected |")
    a("")

    a("## ⚙️ Recommended Parameters")
    a("")
    a("```python")
    a("llm = Llama(")
    a(f"    model_path={lp.model_path!r},")
    a(f"    n_threads={lp.n_threads},")
    a(f"    n_threads_batch={lp.n_threads_batch},")
    a(f"    n_ctx={lp.n_ctx},")
    a(f"    n_batch={lp.n_batch},")
    a(f"    n_gpu_layers={lp.n_gpu_layers},")
    a(f"    use_mmap={lp.use_mmap},")
    a(f"    use_mlock={lp.use_mlock},")
    a(f"    verbose={lp.verbose},")
    a(")")
    a("```")
    a("")

    a("## 🧠 Reasoning")
    a("")
    labels = {
        "params_b": "Parameter count",
        "layers": "Layer count",
        "n_threads": "`n_threads`",
        "n_threads_batch": "`n_threads_batch`",
        "n_ctx": "`n_ctx`",
        "n_gpu_layers": "`n_gpu_layers`",
        "n_batch": "`n_batch`",
        "use_mmap": "`use_mmap`",
        "use_mlock": "`use_mlock`",
    }
    for key, label in labels.items():
        if key in lp.reasoning:
            a(f"**{label}**  ")
            a(f"{lp.reasoning[key]}")
            a("")

    return "\n".join(lines)


def ask_yes_no(prompt: str) -> bool:
    try:
        return input(prompt + " [y/N] ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hardware-aware Llama() constructor parameters from model weight + quant."
    )
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--weight-gb", type=float, help="GGUF file size in GB")
    size_group.add_argument("--params-b", type=float, help="Model parameter count in billions")

    parser.add_argument("--quant", required=True, help=f"Quant type, one of: {', '.join(sorted(QUANT_BPW))}")
    parser.add_argument("--model-path", default="model.gguf", help="Path to pass as model_path")
    parser.add_argument("--context", type=int, default=4096, help="Desired n_ctx")
    parser.add_argument("--layers", type=int, default=None, help="Exact total layer count (improves accuracy)")
    parser.add_argument("--sample-seconds", type=float, default=0,
                         help="Average available RAM over N seconds (10-30 recommended) instead of one snapshot")
    parser.add_argument("--verbose-llama", action="store_true", help="Set verbose=True in the Llama() call")
    parser.add_argument("--json-out", default="llama_params.json")
    parser.add_argument("--md-out", default="llama_params_report.md")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--write-md", action="store_true")
    args = parser.parse_args()

    quant = args.quant.upper()
    if quant not in QUANT_BPW:
        sys.exit(f"Unknown quant '{args.quant}'. Known: {', '.join(sorted(QUANT_BPW))}")

    if args.sample_seconds > 0:
        clamped = max(10, min(30, args.sample_seconds))
        print(f"📊 Sampling available RAM over {clamped:.0f}s for a stable reading…")
    else:
        print("🔍 Detecting host (instant snapshot — pass --sample-seconds 10-30 to average over time)…")

    try:
        result = recommend(
            weight_gb=args.weight_gb,
            params_b=args.params_b,
            quant=quant,
            model_path=args.model_path,
            context=args.context,
            total_layers=args.layers,
            sample_seconds=args.sample_seconds,
            verbose_llama=args.verbose_llama,
        )
    except ValueError as e:
        sys.exit(str(e))

    host, lp, weight_gb = result.host, result.params, result.weight_gb

    print_report(host, lp, weight_gb, quant)

    json_path = Path(args.json_out)
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"✅  JSON → {json_path}")

    write_md = args.write_md
    if not write_md and not args.no_interactive:
        write_md = ask_yes_no("\n📄 Save a human-readable Markdown report?")
    if write_md:
        md_path = Path(args.md_out)
        md_path.write_text(generate_markdown(host, lp, weight_gb, quant), encoding="utf-8")
        print(f"✅  Markdown → {md_path}")


if __name__ == "__main__":
    main()