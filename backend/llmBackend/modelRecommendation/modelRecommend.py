from __future__ import annotations
_W = 'linux_cache_gb'
_V = 'cpu_compute_gb'
_U = 'kv_dtype'
_T = 'kv_cache_gb'
_S = 'rocm-smi'
_R = 'IQ2_XS'
_Q = 'fp16'
_P = 'Linux'
_O = 'Darwin'
_N = 'nvidia-smi'
_M = 'REFERENCE'
_L = 'Q4_K_M'
_K = 'Q3_K_M'
_J = 'Q3_K_S'
_I = False
_H = 'System RAM (CPU-only)'
_G = 'GPU VRAM'
_F = 'Apple Unified Memory'
_E = 'QUALITY'
_D = 'FAST'
_C = 'ULTRA_FAST'
_B = 'BALANCED'
_A = True
import argparse, json, math, platform, re, shutil, subprocess, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
try:
    import psutil
except ImportError:
    sys.exit('psutil is required:  pip install psutil')
DEFAULT_CONTEXT_K = 4
DEFAULT_KV_BITS = 16
MIN_PRACTICAL_TPS = 1.5
TOKENIZER_OVERHEAD_GB = 0.2
LLAMA_N_BATCH = 512
_COMFORT_HEADROOM_GB = {_G: 0.5, _F: 1.5, _H: 2.0}
COMFORT_RATIO = 0.12
BROWSER_OVERHEAD_GB = 2.5
_GPU_EFFICIENCY = 0.9
_RAM_EFFICIENCY = 0.65
_APPLE_EFFICIENCY = 0.78

@dataclass
class Quant:
    name: str
    bpw: float
    tier: str
    quality_note: str
    factor: float = field(init=_I)

    def __post_init__(A):
        A.factor = A.bpw / 8.0
QUANTS = [Quant('IQ1_S', 1.56, _C, '1-bit iMatrix; severe degradation, last resort'), Quant('IQ2_XXS', 2.06, _C, '2-bit iMatrix; smallest practical GGUF'), Quant(_R, 2.31, _C, '2-bit iMatrix XS'), Quant('IQ2_M', 2.57, _C, '2-bit iMatrix M; approaching Q3 quality'), Quant('Q2_K', 2.63, _C, '2-bit k-quant; fast but noticeably lossy'), Quant('IQ3_XXS', 3.06, _D, '3-bit iMatrix XXS'), Quant('IQ3_XS', 3.3, _D, '3-bit iMatrix XS'), Quant(_J, 3.5, _D, '3-bit k-quant small'), Quant(_K, 3.91, _D, '3-bit k-quant medium; good speed/size ratio'), Quant('Q3_K_L', 4.27, _D, '3-bit k-quant large'), Quant('IQ4_XS', 4.25, _B, '4-bit iMatrix XS; best size/quality among 4-bit'), Quant('IQ4_NL', 4.5, _B, '4-bit iMatrix NL'), Quant('Q4_0', 4.55, _B, '4-bit legacy; slightly behind Q4_K_M'), Quant('Q4_K_S', 4.37, _B, '4-bit k-quant small'), Quant(_L, 4.85, _B, '4-bit k-quant medium; community sweet-spot'), Quant('Q4_K_L', 4.9, _B, '4-bit k-quant large'), Quant('Q5_K_S', 5.54, _E, '5-bit k-quant small'), Quant('Q5_K_M', 5.68, _E, '5-bit k-quant medium; near-lossless'), Quant('Q6_K', 6.57, _E, '6-bit k-quant; very close to FP16'), Quant('Q8_0', 8.5, _E, '8-bit; indistinguishable from FP16 in practice'), Quant('F16', 16.0, _M, 'Full FP16; reference quality, enormous VRAM cost')]
QUANT_BY_NAME = {A.name: A for A in QUANTS}

@dataclass
class ModelArch:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int

    @property
    def num_heads(self):
        return self.hidden_size // self.head_dim

@dataclass
class ModelFamily:
    params_b: float
    label: str
    context_k: int
    use_case: str
    families: list[str]
    min_quant: str
    arch: ModelArch
MODEL_FAMILIES = [ModelFamily(0.5, '0.5B', 4, 'Nano / edge device', ['Qwen2.5-0.5B', 'SmolLM-360M'], _L, ModelArch(24, 2, 64, 896, 151936)), ModelFamily(1.0, '1B', 8, 'Tiny assistant, code completion', ['Llama-3.2-1B', 'Qwen2.5-1.5B'], _L, ModelArch(16, 8, 64, 2048, 128256)), ModelFamily(3.0, '3B', 8, 'Light chat / on-device assistant', ['Llama-3.2-3B', 'Phi-3.5-mini', 'Qwen2.5-3B'], _L, ModelArch(28, 8, 128, 3072, 128256)), ModelFamily(7.0, '7B', 8, 'General assistant (sweet-spot)', ['Mistral-7B-v0.3', 'Llama-3.1-8B', 'Qwen2.5-7B', 'Gemma-2-9B'], _K, ModelArch(32, 8, 128, 4096, 128256)), ModelFamily(14.0, '14B', 16, 'Strong reasoning & coding', ['Qwen2.5-14B', 'Phi-4-14B', 'Gemma-2-12B'], _K, ModelArch(48, 8, 128, 5120, 151936)), ModelFamily(22.0, '22B', 16, 'Near-GPT-3.5 quality', ['Qwen2.5-22B', 'Mistral-Small-22B'], _K, ModelArch(48, 8, 128, 6144, 32000)), ModelFamily(32.0, '32B', 32, 'Instruction-tuned powerhouse', ['Qwen2.5-32B', 'Mistral-Small-3.1-32B'], _J, ModelArch(64, 8, 128, 5120, 151936)), ModelFamily(47.0, '47B', 32, 'MoE efficiency (Mixtral-8x7B tier)', ['Mixtral-8x7B'], _J, ModelArch(32, 8, 128, 4096, 32000)), ModelFamily(70.0, '70B', 64, 'Frontier open-source quality', ['Llama-3.1-70B', 'Qwen2.5-72B', 'Nemotron-70B'], _J, ModelArch(80, 8, 128, 8192, 128256)), ModelFamily(123.0, '123B', 128, 'Very large; top open-weight quality', ['Mistral-Large-2', 'Mistral-Large-Instruct-2407'], 'Q2_K', ModelArch(88, 8, 128, 12288, 32768)), ModelFamily(236.0, '236B', 64, 'Huge MoE (Mixtral-8x22B tier)', ['Mixtral-8x22B'], _R, ModelArch(56, 8, 128, 6144, 32000)), ModelFamily(405.0, '405B', 128, 'Largest open-weight (Llama-3.1-405B)', ['Llama-3.1-405B'], 'IQ1_S', ModelArch(126, 8, 128, 16384, 128256))]

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_gb: float
    driver_version: str = ''
    cuda_version: str = ''

def _parse_nvidia_smi():
    try:
        D = subprocess.check_output([_N, '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader,nounits'], text=_A, stderr=subprocess.DEVNULL)
        B = []
        for E in D.strip().splitlines():
            A = [A.strip() for A in E.split(',')]
            if len(A) < 4:
                continue
            B.append(GPUInfo(index=int(A[0]), name=A[1], vram_gb=round(float(A[2]) / 1024, 2), driver_version=A[3]))
        try:
            F = subprocess.check_output([_N], text=_A, stderr=subprocess.DEVNULL)
            C = re.search('CUDA Version:\\s*([\\d.]+)', F)
            if C:
                for G in B:
                    G.cuda_version = C.group(1)
        except Exception:
            pass
        return B
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

def _parse_rocm():
    try:
        C = subprocess.check_output([_S, '--showmeminfo', 'vram', '--json'], text=_A, stderr=subprocess.DEVNULL)
        D = json.loads(C)
        A = []
        for E, B in D.items():
            if 'card' in E.lower():
                F = int(B.get('VRAM Total Memory (B)', 0))
                A.append(GPUInfo(index=len(A), name=B.get('Card series', 'AMD GPU'), vram_gb=round(F / 1024 ** 3, 2)))
        return A
    except Exception:
        return []

def _parse_metal():
    D = 'Apple Silicon'
    C = 'SPHardwareDataType'
    if platform.system() != _O:
        return []
    try:
        E = subprocess.check_output(['system_profiler', C, '-json'], text=_A, stderr=subprocess.DEVNULL)
        F = json.loads(E)
        B = F.get(C, [{}])[0]
        G = B.get('chip_type', B.get('cpu_type', D))
        A = psutil.virtual_memory().total / 1024 ** 3
        return [GPUInfo(index=0, name=G, vram_gb=round(A * 0.75, 1))]
    except Exception:
        if platform.processor() == 'arm':
            A = psutil.virtual_memory().total / 1024 ** 3
            return [GPUInfo(index=0, name=D, vram_gb=round(A * 0.75, 1))]
        return []

@dataclass
class SystemInfo:
    ram_gb: float
    os: str
    cpu: str
    cpu_cores: int
    cpu_freq_ghz: float
    gpus: list[GPUInfo]
    has_avx2: bool
    has_avx512: bool
    is_apple_silicon: bool

    @property
    def total_vram_gb(self):
        return sum((A.vram_gb for A in self.gpus))

    @property
    def best_gpu(self):
        return max(self.gpus, key=lambda g: g.vram_gb) if self.gpus else None

    @property
    def gpu_names(self):
        return ', '.join((A.name for A in self.gpus)) if self.gpus else 'None'

def detect_system():
    H = psutil.virtual_memory()
    I = round(H.total / 1024 ** 3, 1)
    B = psutil.cpu_freq()
    J = round((B.max or B.current) / 1000, 2) if B else 0.0
    F = platform.processor() == 'arm' and platform.system() == _O
    C = D = _I
    try:
        if platform.system() == _P:
            G = Path('/proc/cpuinfo').read_text()
            C = ' avx2 ' in G
            D = ' avx512f ' in G
        elif platform.system() == _O:
            E = subprocess.check_output(['sysctl', '-a', 'machdep.cpu.leaf7_features'], text=_A, stderr=subprocess.DEVNULL)
            C = 'AVX2' in E
            D = 'AVX512F' in E
        elif platform.system() == 'Windows':
            E = subprocess.check_output(['wmic', 'cpu', 'get', 'Caption'], text=_A, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    A = []
    if shutil.which(_N):
        A = _parse_nvidia_smi()
    elif shutil.which(_S):
        A = _parse_rocm()
    elif F:
        A = _parse_metal()
    return SystemInfo(ram_gb=I, os=f'{platform.system()} {platform.release()}', cpu=platform.processor() or platform.machine(), cpu_cores=psutil.cpu_count(logical=_I) or psutil.cpu_count() or 1, cpu_freq_ghz=J, gpus=A, has_avx2=C, has_avx512=D, is_apple_silicon=F)

@dataclass
class MemoryBudget:
    source: str
    total_gb: float
    usable_gb: float
    note: str

def compute_budget(sys):
    B = sys
    if B.gpus and (not B.is_apple_silicon):
        A = B.total_vram_gb
        return MemoryBudget(source=_G, total_gb=round(A, 2), usable_gb=round(A * _GPU_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB, note=B.gpu_names)
    if B.is_apple_silicon:
        A = B.ram_gb
        return MemoryBudget(source=_F, total_gb=round(A, 2), usable_gb=round(A * _APPLE_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB, note='Apple Silicon – CPU and GPU share the same physical RAM pool')
    A = B.ram_gb
    return MemoryBudget(source=_H, total_gb=round(A, 2), usable_gb=round(A * _RAM_EFFICIENCY, 2) - BROWSER_OVERHEAD_GB, note='No GPU detected – llama.cpp will run in CPU-only mode')

@dataclass
class MemoryOverhead:
    kv_cache_gb: float
    kv_context_k: int
    kv_dtype_bytes: int
    cpu_compute_gb: float
    linux_cache_gb: float
    tokenizer_gb: float

    @property
    def total_gb(self):
        A = self
        return round(A.kv_cache_gb + A.cpu_compute_gb + A.linux_cache_gb + A.tokenizer_gb, 3)

    def to_dict(A):
        B = _Q if A.kv_dtype_bytes == 2 else 'int8'
        return {_T: round(A.kv_cache_gb, 3), 'kv_context_k': A.kv_context_k, _U: B, _V: round(A.cpu_compute_gb, 3), _W: round(A.linux_cache_gb, 3), 'tokenizer_gb': round(A.tokenizer_gb, 3), 'total_overhead_gb': A.total_gb}

def _measure_linux_cache_pressure_gb():
    try:
        A = psutil.virtual_memory()
        B = (getattr(A, 'cached', 0) + getattr(A, 'buffers', 0)) / 1024 ** 3
        C = B * 0.1
        return round(max(0.3, C), 2)
    except Exception:
        return 0.5

def compute_overhead(model, context_k, kv_dtype_bytes, budget):
    E = budget
    D = kv_dtype_bytes
    C = context_k
    A = model.arch
    F = C * 1024
    G = E.source == _G
    H = E.source == _F
    N = platform.system() == _P
    J = 2 * A.num_layers * A.num_kv_heads * A.head_dim * F * D
    K = J / 1024 ** 3
    B = 0.0
    if not G and (not H):
        L = A.num_heads * LLAMA_N_BATCH * F * 4
        M = 3 * A.hidden_size * LLAMA_N_BATCH * 4
        B = max(0.1, (L + M) / 1024 ** 3)
        B = round(B, 3)
    I = 0.0
    if platform.system() == _P and (not G) and (not H):
        I = _measure_linux_cache_pressure_gb()
    return MemoryOverhead(kv_cache_gb=round(K, 3), kv_context_k=C, kv_dtype_bytes=D, cpu_compute_gb=B, linux_cache_gb=I, tokenizer_gb=TOKENIZER_OVERHEAD_GB)
_GPU_BW = [('H200', 4800.0), ('H100 SXM', 3350.0), ('H100 PCIe', 2000.0), ('A100 SXM 80', 2000.0), ('A100 PCIe 80', 1935.0), ('A100', 1555.0), ('A40', 696.0), ('A30', 933.0), ('4090', 1008.0), ('4080 SUPER', 736.0), ('4080', 716.0), ('4070 Ti SUPER', 672.0), ('4070 Ti', 504.0), ('4070 SUPER', 504.0), ('4070', 504.0), ('4060 Ti', 288.0), ('4060', 272.0), ('4090 Laptop', 576.0), ('4080 Laptop', 432.0), ('4070 Laptop', 288.0), ('4060 Laptop', 192.0), ('3090 Ti', 1008.0), ('3090', 936.0), ('3080 Ti', 912.0), ('3080 12GB', 912.0), ('3080', 760.0), ('3070 Ti', 608.0), ('3070', 448.0), ('3060 Ti', 448.0), ('3060', 360.0), ('3080 Laptop', 384.0), ('3070 Laptop', 256.0), ('3060 Laptop', 192.0), ('2080 Ti', 616.0), ('2080 SUPER', 496.0), ('2080', 448.0), ('2070 SUPER', 448.0), ('2070', 448.0), ('2060 SUPER', 448.0), ('2060', 336.0), ('1080 Ti', 484.0), ('1080', 320.0), ('1070 Ti', 256.0), ('1070', 256.0), ('1060', 192.0), ('A6000', 768.0), ('A5000', 768.0), ('A4000', 448.0), ('A2000', 288.0), ('MI300X', 5300.0), ('MI250X', 3277.0), ('MI210', 1600.0), ('RX 7900 XTX', 960.0), ('RX 7900 XT', 800.0), ('RX 7800 XT', 624.0), ('RX 7700 XT', 432.0), ('RX 6900 XT', 512.0), ('RX 6800 XT', 512.0), ('RX 6700 XT', 384.0)]
_APPLE_BW = [('M4 Ultra', 800.0), ('M4 Max', 300.0), ('M4 Pro', 120.0), ('M4', 75.0), ('M3 Ultra', 800.0), ('M3 Max', 300.0), ('M3 Pro', 150.0), ('M3', 100.0), ('M2 Ultra', 800.0), ('M2 Max', 400.0), ('M2 Pro', 200.0), ('M2', 100.0), ('M1 Ultra', 800.0), ('M1 Max', 400.0), ('M1 Pro', 200.0), ('M1', 68.0)]

def _match_bw(name, table, default):
    D = name.upper()
    B, C = (0, default)
    for A, E in table:
        if A.upper() in D and len(A) > B:
            B, C = (len(A), E)
    return C

def _cpu_effective_bw_gb_s(sys_info):
    A = sys_info
    C = A.cpu_freq_ghz
    if A.has_avx512:
        B = 90.0
    elif A.has_avx2 and C >= 4.5:
        B = 65.0
    elif A.has_avx2 and C >= 3.5:
        B = 50.0
    elif A.has_avx2:
        B = 38.0
    else:
        B = 20.0
    if not A.has_avx2:
        B *= 0.5
    return B

def estimate_tps(weights_gb, sys_info, budget):
    E = budget
    D = weights_gb
    A = sys_info
    if D <= 0:
        return 0.0
    if E.source == _G:
        B = A.best_gpu.name if A.best_gpu else ''
        C = _match_bw(B, _GPU_BW, default=400.0)
    elif E.source == _F:
        B = A.gpus[0].name if A.gpus else ''
        C = _match_bw(B, _APPLE_BW, default=68.0)
    else:
        C = _cpu_effective_bw_gb_s(A)
    return round(C / (2.0 * D), 1)

@dataclass
class Recommendation:
    quant: Quant
    model: ModelFamily
    weights_gb: float
    overhead: MemoryOverhead
    fits: bool
    fits_comfortably: bool
    headroom_gb: float
    speed_label: str
    quality_stars: int
    est_tps: float
    is_practical: bool

    @property
    def total_gb(self):
        return round(self.weights_gb + self.overhead.total_gb, 2)
_TIER_STAR = {_C: 1, _D: 2, _B: 3, _E: 4, _M: 5}
_SPEED_TABLE = [(2.5, 'Very Fast'), (3.5, 'Fast'), (5.0, 'Moderate'), (8.5, 'Somewhat Slow'), (99.0, 'Slow')]

def _speed_label(bpw):
    for A, B in _SPEED_TABLE:
        if bpw <= A:
            return B
    return 'Slow'

def build_recommendations(budget, sys_info, context_k, kv_dtype_bytes, min_bpw=0.0):
    H = kv_dtype_bytes
    G = context_k
    A = budget
    M = A.source == _H
    O = _COMFORT_HEADROOM_GB[A.source]
    I = []
    for B in QUANTS:
        if B.bpw < min_bpw:
            continue
        C = None
        for E in MODEL_FAMILIES:
            D = compute_overhead(E, G, H, A)
            N = E.params_b * B.factor + D.total_gb
            if N <= A.usable_gb:
                C = E
        if C is None:
            continue
        F = C.params_b * B.factor
        D = compute_overhead(C, G, H, A)
        J = F + D.total_gb
        K = A.usable_gb - J
        L = estimate_tps(F, sys_info, A)
        I.append(Recommendation(quant=B, model=C, weights_gb=round(F, 2), overhead=D, fits=J <= A.usable_gb, fits_comfortably=K >= _COMFORT_HEADROOM_GB[A.source], headroom_gb=round(K, 2), speed_label=_speed_label(B.bpw), quality_stars=_TIER_STAR.get(B.tier, 3), est_tps=L, is_practical=L >= MIN_PRACTICAL_TPS if M else _A))
    return I

def top_picks(recs):
    B = {}
    for C in (_C, _D, _B, _E, _M):
        A = [A for A in recs if A.quant.tier == C and A.fits]
        if not A:
            continue
        for D in [[A for A in A if A.fits_comfortably and A.is_practical], [A for A in A if A.fits_comfortably], [A for A in A if A.is_practical], A]:
            if D:
                B[C] = max(D, key=lambda r: (r.model.params_b, r.quant.bpw))
                break
    return B

def build_json_output(sys_info, budget, recs, picks, context_k, kv_dtype_bytes):
    M = 'quality_stars'
    L = 'speed'
    K = 'is_practical'
    J = 'est_tps'
    I = 'fits_comfortably'
    H = 'headroom_gb'
    G = 'total_memory_gb'
    F = 'weights_gb'
    E = 'bpw'
    D = 'quant'
    C = kv_dtype_bytes
    B = budget
    A = sys_info

    def N(g):
        A = {'name': g.name, 'vram_gb': g.vram_gb}
        if g.driver_version:
            A['driver'] = g.driver_version
        if g.cuda_version:
            A['cuda'] = g.cuda_version
        return A
    return {'system': {'os': A.os, 'cpu': A.cpu, 'cpu_cores_physical': A.cpu_cores, 'cpu_freq_ghz': A.cpu_freq_ghz, 'ram_gb': A.ram_gb, 'avx2': A.has_avx2, 'avx512': A.has_avx512, 'apple_silicon': A.is_apple_silicon, 'gpus': [N(A) for A in A.gpus]}, 'memory_budget': {'source': B.source, 'total_gb': B.total_gb, 'usable_gb': B.usable_gb, 'note': B.note}, 'inference_settings': {'context_k': context_k, _U: _Q if C == 2 else 'int8', 'kv_dtype_bytes': C}, 'top_picks': {B: {D: A.quant.name, E: A.quant.bpw, 'model': A.model.label, 'params_b': A.model.params_b, F: A.weights_gb, 'overhead': A.overhead.to_dict(), G: A.total_gb, H: A.headroom_gb, I: A.fits_comfortably, J: A.est_tps, K: A.is_practical, L: A.speed_label, M: A.quality_stars, 'example_models': A.model.families[:3], 'use_case': A.model.use_case, 'quality_note': A.quant.quality_note} for B, A in picks.items()}, 'full_quant_table': [{D: A.quant.name, 'tier': A.quant.tier, E: A.quant.bpw, 'best_model': A.model.label, F: A.weights_gb, _T: A.overhead.kv_cache_gb, _V: A.overhead.cpu_compute_gb, _W: A.overhead.linux_cache_gb, G: A.total_gb, I: A.fits_comfortably, H: A.headroom_gb, J: A.est_tps, K: A.is_practical, L: A.speed_label, M: A.quality_stars} for A in recs]}
_STARS = {1: '★☆☆☆☆', 2: '★★☆☆☆', 3: '★★★☆☆', 4: '★★★★☆', 5: '★★★★★'}
_TIER_EMOJI = {_C: '⚡', _D: '🚀', _B: '⚖️', _E: '💎', _M: '🔬'}

def generate_markdown(sys_info, budget, recs, picks, context_k, kv_dtype_bytes):
    G = kv_dtype_bytes
    E = budget
    C = sys_info
    H = []
    A = H.append
    I = E.source == _H
    A('# 🤖 LLM Hardware Compatibility Report')
    A('')
    A('> Auto-generated by `model_recommender.py` — llama.cpp-aware memory model')
    A('')
    A('## 🖥️ System')
    A('')
    A('| Property | Value |')
    A('|----------|-------|')
    A(f'| OS | `{C.os}` |')
    A(f'| CPU | `{C.cpu}` · {C.cpu_cores} physical cores @ {C.cpu_freq_ghz} GHz |')
    A(f'| RAM | **{C.ram_gb} GB** |')
    A(f'| AVX2 / AVX-512 | {C.has_avx2} / {C.has_avx512} |')
    if C.gpus:
        for D in C.gpus:
            K = f' · driver {D.driver_version}' if D.driver_version else ''
            L = f', CUDA {D.cuda_version}' if D.cuda_version else ''
            A(f'| GPU {D.index} | **{D.name}** · {D.vram_gb} GB VRAM{K}{L} |')
    else:
        A('| GPU | None detected (CPU-only mode) |')
    A('')
    if not C.has_avx2 and I:
        A('> ⚠️ **No AVX2 detected.** llama.cpp will use scalar fallback kernels, which are ~2–3× slower than AVX2. Performance estimates are adjusted accordingly.')
        A('')
    M = _Q if G == 2 else 'int8 (q8_0)'
    A('## 💾 Memory Budget')
    A('')
    A('| Item | Value |')
    A('|------|-------|')
    A(f'| Source | {E.source} |')
    A(f'| Physical total | {E.total_gb} GB |')
    A(f'| After OS/driver overhead | **{E.usable_gb} GB** |')
    A(f'| Target context | {context_k}K tokens |')
    A(f'| KV-cache dtype | {M} ({G} bytes/elem) |')
    A(f'| Note | {E.note} |')
    A('')
    A('> Per-model costs (KV cache, compute buffer, tokenizer) are deducted per recommendation below.')
    A('')
    A('## 🏆 Top Picks (one per tier)')
    A('')
    for J, B in picks.items():
        N = _TIER_EMOJI.get(J, '•')
        O = ' ⚠️ *impractical on CPU*' if not B.is_practical else ''
        A(f"### {N} {J.replace('_', ' ').title()}{O}")
        A(f'| Field | Value |')
        A(f'|-------|-------|')
        A(f'| Quant | `{B.quant.name}` ({B.quant.bpw:.2f} bpw) |')
        A(f'| Model | **{B.model.label}** ({B.model.params_b}B params) |')
        A(f'| Weights | {B.weights_gb} GB |')
        A(f'| KV cache @ {B.overhead.kv_context_k}K | {B.overhead.kv_cache_gb:.3f} GB |')
        if B.overhead.cpu_compute_gb > 0:
            A(f'| CPU compute buffer | {B.overhead.cpu_compute_gb:.3f} GB |')
        if B.overhead.linux_cache_gb > 0:
            A(f'| Linux page-cache reserve | {B.overhead.linux_cache_gb:.3f} GB |')
        A(f'| Tokenizer overhead | {B.overhead.tokenizer_gb:.2f} GB |')
        A(f'| **Total memory** | **{B.total_gb} GB** |')
        A(f"| Headroom | {B.headroom_gb:.2f} GB {('✅' if B.fits_comfortably else '⚠️')} |")
        A(f'| Est. decode speed | ~{B.est_tps} t/s |')
        A(f"| Quality | {_STARS.get(B.quality_stars, '')} |")
        A(f'| Use case | {B.model.use_case} |')
        A(f"| Example models | {', '.join(B.model.families[:3])} |")
        A(f'| Notes | {B.quant.quality_note} |')
        A('')
    A('## 📊 Full Quant × Model Table')
    A('')
    A('| Quant | Tier | BPW | Model | Weights | KV$ | CPU buf | Total | Comfortable | ~t/s | Quality |')
    A('|-------|------|-----|-------|---------|-----|---------|-------|:-----------:|------|---------|')
    for B in recs:
        P = '✅' if B.fits_comfortably else '⚠️'
        Q = '' if B.is_practical else ' 🐢'
        A(f"| `{B.quant.name}` | {B.quant.tier} | {B.quant.bpw:.2f} | {B.model.label} | {B.weights_gb:.1f} | {B.overhead.kv_cache_gb:.2f} | {B.overhead.cpu_compute_gb:.2f} | {B.total_gb:.1f} | {P} | {B.est_tps}{Q} | {_STARS.get(B.quality_stars, '')} |")
    A('')
    A('*KV$* = KV-cache GB at target context. *CPU buf* = ggml compute buffer (0 on GPU/Apple). *🐢* = estimated <1.5 t/s on CPU.')
    A('')
    A('## 💡 llama.cpp Tips')
    A('')
    R = ['`Q4_K_M` is the community sweet-spot: great quality, reasonable size.', '`IQ4_XS` is slightly smaller than `Q4_K_M` with minimal quality loss (requires iMatrix).', '`Q8_0` is the highest-quality quant worth running; `F16` is overkill for inference.', 'Reduce `--ctx-size` to shrink the KV cache linearly — halving context halves KV RAM.', 'Use `--kv-cache-type q8_0` to halve KV cache size with ~0.5% quality loss.']
    if I:
        F = [f'Set `--threads {C.cpu_cores}` (physical cores only; hyperthreads hurt throughput).', 'Add `--flash-attn` to dramatically reduce the ggml compute buffer at long contexts.', 'Add `--no-mmap` on Linux if you experience stuttering from page-cache eviction.', 'Prompt processing (prefill) benefits from `--n-batch 512`; generation speed does not.']
        if not C.has_avx2:
            F.insert(0, '⚠️ Compile llama.cpp with `-DLLAMA_AVX2=ON` if your CPU actually supports it — the packaged binary may not have it enabled.')
    elif E.source == _F:
        F = ['Use `--n-gpu-layers 999` to offload all layers to Metal for maximum speed.', 'Add `--flash-attn` — Metal flash-attention is well-optimised on M-series.', 'Avoid `--no-mmap`; mmap is efficient on macOS due to unified memory.']
    else:
        F = ['Use `--n-gpu-layers 999` to fully offload the model to VRAM.', 'Add `--flash-attn` to reduce attention memory and allow longer contexts.', 'With multiple GPUs, `--tensor-split` splits layers; single large GPU is usually faster.', 'Compile llama.cpp with `cmake -DGGML_CUDA=ON` for best performance.']
    for S in R + F:
        A(f'- {S}')
    A('')
    return '\n'.join(H)

def ask_yes_no(prompt):
    try:
        return input(prompt + ' [y/N] ').strip().lower() in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return _I

def main():
    O = 'utf-8'
    N = 'store_true'
    F = '─'
    B = argparse.ArgumentParser(description='Detect hardware and recommend GGUF quant + model for llama.cpp.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    B.add_argument('--json-out', default='quant_model_map.json')
    B.add_argument('--md-out', default='quant_model_report.md')
    B.add_argument('--min-bpw', type=float, default=0.0, help='Exclude quants below this BPW (e.g. 3.0 skips 1–2 bit quants)')
    B.add_argument('--context', type=int, default=DEFAULT_CONTEXT_K, help='Target inference context in K tokens (e.g. 4 = 4096 tokens)')
    B.add_argument('--kv-bits', type=int, default=DEFAULT_KV_BITS, choices=[8, 16], help='KV-cache dtype: 16 = fp16 (default), 8 = int8 (--kv-cache-type q8_0)')
    B.add_argument('--no-interactive', action=N, help='Skip interactive prompts; suppress Markdown output')
    B.add_argument('--write-md', action=N, help='Always write Markdown report without prompting')
    A = B.parse_args()
    G = A.kv_bits // 8
    print('🔍 Detecting hardware…', flush=_A)
    E = detect_system()
    C = compute_budget(E)
    print(f'   {C.source}: {C.total_gb} GB total → {C.usable_gb} GB usable')
    if E.gpus:
        print(f'   GPU(s): {E.gpu_names}')
    print(f'   Context: {A.context}K tokens · KV dtype: fp{A.kv_bits}')
    print('', flush=_A)
    H = build_recommendations(C, E, A.context, G, A.min_bpw)
    I = top_picks(H)
    K = build_json_output(E, C, H, I, A.context, G)
    L = Path(A.json_out)
    L.write_text(json.dumps(K, indent=2), encoding=O)
    print(json.dumps(K, indent=2))
    print(f'\n✅  JSON  → {L}', flush=_A)
    print('\n' + F * 70)
    print(f"  {'QUICK SUMMARY':^66}")
    print(F * 70)
    print(f'  Memory source : {C.source}')
    print(f'  Usable budget : {C.usable_gb} GB')
    print(f'  Context       : {A.context}K tokens · KV fp{A.kv_bits}')
    print(F * 70)
    P = f"  {'Tier':<14}  {'Quant':<12}  {'Model':<7}  {'Total GB':>8}  {'Headroom':>9}  {'~t/s':>6}"
    print(P)
    print('  ' + F * 66)
    for Q, D in I.items():
        R = '' if D.is_practical else ' 🐢'
        S = '✅' if D.fits_comfortably else '⚠️ '
        print(f'  {Q:<14}  {D.quant.name:<12}  {D.model.label:<7}  {D.total_gb:>7.1f} GB  {D.headroom_gb:>6.1f} GB {S}  {D.est_tps:>5.0f}{R}')
    print(F * 70)
    J = A.write_md
    if not J and (not A.no_interactive):
        J = ask_yes_no('\n📄 Save a human-readable Markdown report?')
    if J:
        M = Path(A.md_out)
        M.write_text(generate_markdown(E, C, H, I, A.context, G), encoding=O)
        print(f'✅  Markdown → {M}')
if __name__ == '__main__':
    main()
