from __future__ import annotations
_Q='model.gguf'
_P='nvidia-smi'
_O='Darwin'
_N='\n'
_M='layers'
_L='params_b'
_K='use_mmap'
_J='n_batch'
_I='n_ctx'
_H='n_threads_batch'
_G='n_threads'
_F=False
_E=', '
_D='use_mlock'
_C='n_gpu_layers'
_B=True
_A=None
import argparse,json,platform,shutil,subprocess,sys
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Optional
try:import psutil
except ImportError:raise ImportError('psutil is required for llama_param_calculator:  pip install psutil')from _A
__all__=['QUANT_BPW','GPUInfo','HostInfo','LlamaParams','RecommendationResult','detect_host','calculate_params','recommend','generate_markdown']
QUANT_BPW={'IQ1_S':1.56,'IQ2_XXS':2.06,'IQ2_XS':2.31,'IQ2_M':2.57,'Q2_K':2.63,'IQ3_XXS':3.06,'IQ3_XS':3.3,'Q3_K_S':3.5,'Q3_K_M':3.91,'Q3_K_L':4.27,'IQ4_XS':4.25,'IQ4_NL':4.5,'Q4_0':4.55,'Q4_K_S':4.37,'Q4_K_M':4.85,'Q4_K_L':4.9,'Q5_K_S':5.54,'Q5_K_M':5.68,'Q6_K':6.57,'Q8_0':8.5,'F16':16.}
GGUF_OVERHEAD_FRACTION=.04
@dataclass
class GPUInfo:name:str;vram_gb:float
@dataclass
class HostInfo:
	ram_gb:float;available_ram_gb:float;physical_cores:int;logical_cores:int;has_avx2:bool;is_apple_silicon:bool;gpus:list[GPUInfo]
	@property
	def total_vram_gb(self):return sum(A.vram_gb for A in self.gpus)
	@property
	def gpu_names(self):return _E.join(A.name for A in self.gpus)if self.gpus else'None'
def _nvidia_gpus():
	try:
		B=subprocess.check_output([_P,'--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=_B,stderr=subprocess.DEVNULL);A=[]
		for C in B.strip().splitlines():D,E=[A.strip()for A in C.split(',')];A.append(GPUInfo(name=D,vram_gb=round(float(E)/1024,2)))
		return A
	except Exception:return[]
def _apple_gpu(ram_gb):
	B='SPHardwareDataType'
	if platform.system()!=_O or platform.processor()!='arm':return[]
	A='Apple Silicon'
	try:C=subprocess.check_output(['system_profiler',B,'-json'],text=_B,stderr=subprocess.DEVNULL);D=json.loads(C).get(B,[{}])[0];A=D.get('chip_type',A)
	except Exception:pass
	return[GPUInfo(name=A,vram_gb=round(ram_gb*.75,1))]
def detect_host(sample_seconds=.0):
	D=sample_seconds;E=psutil.virtual_memory();F=round(E.total/1024**3,2)
	if D>0:
		import time;A=[];G=max(2,int(D/.5))
		for J in range(G):
			A.append(psutil.virtual_memory().available/1024**3)
			if J<G-1:time.sleep(.5)
		H=round(sum(A)/len(A),2)
	else:H=round(E.available/1024**3,2)
	B=_F
	try:
		if platform.system()=='Linux':B=' avx2 'in Path('/proc/cpuinfo').read_text()
		elif platform.system()==_O:K=subprocess.check_output(['sysctl','-a','machdep.cpu.leaf7_features'],text=_B,stderr=subprocess.DEVNULL);B='AVX2'in K
	except Exception:pass
	I=platform.processor()=='arm'and platform.system()==_O;C=[]
	if shutil.which(_P):C=_nvidia_gpus()
	elif I:C=_apple_gpu(F)
	return HostInfo(ram_gb=F,available_ram_gb=H,physical_cores=psutil.cpu_count(logical=_F)or psutil.cpu_count()or 1,logical_cores=psutil.cpu_count(logical=_B)or 1,has_avx2=B,is_apple_silicon=I,gpus=C)
@dataclass
class LlamaParams:
	model_path:str;n_threads:int;n_threads_batch:int;n_ctx:int;n_batch:int;n_gpu_layers:int;use_mmap:bool;use_mlock:bool;verbose:bool;reasoning:dict[str,str]
	def to_constructor_kwargs(A):return{'model_path':A.model_path,_G:A.n_threads,_H:A.n_threads_batch,_I:A.n_ctx,_J:A.n_batch,_C:A.n_gpu_layers,_K:A.use_mmap,_D:A.use_mlock,'verbose':A.verbose}
def _estimate_total_layers(params_b):A=[(.5,24),(1.,16),(3.,28),(7.,32),(14.,48),(22.,48),(32.,64),(47.,32),(7e1,80),(123.,88),(236.,56),(405.,126)];B=min(A,key=lambda a:abs(a[0]-params_b));return B[1]
def calculate_params(model_path,weight_gb,quant,host,context,params_b=_A,total_layers=_A,verbose_flag=_F):
	U=total_layers;T=context;N=quant;G=params_b;D=weight_gb;B=host;A={};O=QUANT_BPW.get(N.upper())
	if O is _A:raise ValueError(f"Unknown quant '{N}'. Known: {_E.join(sorted(QUANT_BPW))}")
	if G is _A:G=D/(O/8.*(1+GGUF_OVERHEAD_FRACTION));A[_L]=f"Inferred {G:.2f}B params from {D} GB at {N} ({O} bpw, +{GGUF_OVERHEAD_FRACTION*100:.0f}% GGUF overhead)."
	else:A[_L]=f"Given explicitly: {G}B params."
	C=U or _estimate_total_layers(G)
	if U:A[_M]=f"Given explicitly: {C} layers."
	else:A[_M]=f"Estimated {C} layers from {G:.1f}B params via nearest-known-model anchor (no --layers supplied; pass it for exact results)."
	V=max(1,B.physical_cores);A[_G]=f"= physical core count ({B.physical_cores}). Logical core count is {B.logical_cores}; hyperthreads are excluded because llama.cpp's matmul kernels are bandwidth-bound, not execution-port-bound — SMT siblings mostly contend rather than add throughput.";Z=V;A[_H]='= n_threads (prompt-processing scales the same way as generation).';P=T;A[_I]=f"Using requested context: {T} tokens.";W=bool(B.gpus);Q=B.total_vram_gb;R=D/max(1,C)
	if W:
		X=.85 if not B.is_apple_silicon else .75;I=Q*X;J=C*(P/1024)*.05;K=.3;a=max(.0,I-J-K);F=int(a/R)if R>0 else 0;F=max(0,min(C,F))
		if F>=C:E=-1;A[_C]=f"-1 (all {C} layers). Estimated VRAM need ≈ {D+J+K:.2f} GB (weights {D:.2f} + KV-cache ≈{J:.2f} + compute buffer ≈{K:.2f}) fits in {I:.2f} GB usable VRAM ({Q:.2f} GB physical × {X:.0%} driver/headroom factor) on {B.gpu_names}."
		elif F>0:E=F;A[_C]=f"{F} of {C} layers (partial offload). Full model + KV-cache + compute buffer needs ≈{D+J+K:.2f} GB but only {I:.2f} GB usable VRAM is available on {B.gpu_names} ({Q:.2f} GB physical). Remaining {C-F} layers run on CPU; expect a real slowdown from the CPU↔GPU handoff at the split boundary, not just blended bandwidth."
		else:E=0;A[_C]=f"0 — even a single layer (≈{R:.3f} GB) plus KV-cache/compute overhead doesn't comfortably fit in {I:.2f} GB usable VRAM on {B.gpu_names}. Running CPU-only."
	else:E=0;A[_C]='0 — no GPU detected; CPU-only inference.'
	L=max(.5,B.available_ram_gb-D*(0 if E==-1 else 1))
	if L>=4:H=1024
	elif L>=2:H=512
	elif L>=1:H=256
	else:H=128
	A[_J]=f"{H} — chosen from ≈{L:.2f} GB RAM headroom left after {"weights remain on GPU"if E==-1 else"CPU-resident weights"} ({B.available_ram_gb:.2f} GB available RAM measured). Larger batches speed up prompt processing but raise compute-buffer RAM use roughly linearly; this caps it before swapping risk.";b=_B;A[_K]='True — lets the OS lazily page in the GGUF file and share pages across processes; near-instant load time and lower peak RSS. Almost always the right default.';c=C*(P/1024)*.05 if not W else .0;M=D+c;S=B.ram_gb*.2;Y=E!=-1 and B.ram_gb-M>=S;d=Y
	if E==-1:A[_D]='False — model is fully GPU-resident; locking host RAM pages has no benefit.'
	elif Y:A[_D]=f"True — estimated working set ≈{M:.2f} GB leaves {B.ram_gb-M:.2f} GB free out of {B.ram_gb:.2f} GB total RAM, above the {S:.2f} GB safety margin. Locking prevents the model from being swapped out under memory pressure, trading flexibility for consistent latency."
	else:A[_D]=f"False — estimated working set ≈{M:.2f} GB would leave less than {S:.2f} GB free out of {B.ram_gb:.2f} GB total RAM if locked. Locking here risks starving the OS and other processes; let pages be swappable instead."
	return LlamaParams(model_path=model_path,n_threads=V,n_threads_batch=Z,n_ctx=P,n_batch=H,n_gpu_layers=E,use_mmap=b,use_mlock=d,verbose=verbose_flag,reasoning=A)
@dataclass
class RecommendationResult:
	host:HostInfo;params:LlamaParams;weight_gb:float;quant:str
	def to_constructor_kwargs(A):return A.params.to_constructor_kwargs()
	def to_dict(A):return{'host':asdict(A.host),'input':{'weight_gb':round(A.weight_gb,3),'quant':A.quant},'constructor_kwargs':A.params.to_constructor_kwargs(),'reasoning':A.params.reasoning}
	def to_markdown(A):return generate_markdown(A.host,A.params,A.weight_gb,A.quant)
def recommend(*,weight_gb=_A,params_b=_A,quant,model_path=_Q,context=4096,total_layers=_A,host=_A,sample_seconds=0,verbose_llama=_F):
	D=params_b;C=host;B=weight_gb;A=quant
	if(B is _A)==(D is _A):raise ValueError('Pass exactly one of weight_gb or params_b, not both/neither.')
	A=A.upper()
	if A not in QUANT_BPW:raise ValueError(f"Unknown quant '{A}'. Known: {_E.join(sorted(QUANT_BPW))}")
	if C is _A:C=detect_host(sample_seconds=sample_seconds)
	if B is _A:E=QUANT_BPW[A];B=D*(E/8.)*(1+GGUF_OVERHEAD_FRACTION)
	F=calculate_params(model_path=model_path,weight_gb=B,quant=A,host=C,context=context,params_b=D,total_layers=total_layers,verbose_flag=verbose_llama);return RecommendationResult(host=C,params=F,weight_gb=B,quant=A)
def print_report(host,lp,weight_gb,quant):
	C='─';B=host;A=lp;print(_N+C*70);print(f"  {"HOST":^66}");print(C*70);print(f"  RAM           : {B.ram_gb} GB total, {B.available_ram_gb} GB available");print(f"  CPU cores     : {B.physical_cores} physical / {B.logical_cores} logical");print(f"  AVX2          : {B.has_avx2}");print(f"  GPU(s)        : {B.gpu_names}"+(f"  ({B.total_vram_gb:.1f} GB VRAM total)"if B.gpus else''));print(_N+C*70);print(f"  {"RECOMMENDED Llama() PARAMETERS":^66}");print(C*70);E=f"""llm = Llama(
    model_path={A.model_path!r},
    n_threads={A.n_threads},
    n_threads_batch={A.n_threads_batch},
    n_ctx={A.n_ctx},
    n_batch={A.n_batch},
    n_gpu_layers={A.n_gpu_layers},
    use_mmap={A.use_mmap},
    use_mlock={A.use_mlock},
    verbose={A.verbose},
)""";print(E);print(_N+C*70);print(f"  {"REASONING":^66}");print(C*70)
	for D in[_L,_M,_G,_H,_I,_C,_J,_K,_D]:
		if D in A.reasoning:print(f"\n  • {D}:");print(f"      {A.reasoning[D]}")
	print()
def generate_markdown(host,lp,weight_gb,quant):
	C=host;B=lp;D=[];A=D.append;A('# 🦙 llama.cpp Parameter Recommendation');A('');A(f"> Calculated for a **{weight_gb} GB** model at **{quant}** quantization.");A('');A('## 🖥️ Host');A('');A('| Property | Value |');A('|----------|-------|');A(f"| RAM | {C.ram_gb} GB total, {C.available_ram_gb} GB available |");A(f"| CPU cores | {C.physical_cores} physical / {C.logical_cores} logical |");A(f"| AVX2 | {C.has_avx2} |")
	if C.gpus:
		for E in C.gpus:A(f"| GPU | {E.name} — {E.vram_gb} GB VRAM |")
	else:A('| GPU | None detected |')
	A('');A('## ⚙️ Recommended Parameters');A('');A('```python');A('llm = Llama(');A(f"    model_path={B.model_path!r},");A(f"    n_threads={B.n_threads},");A(f"    n_threads_batch={B.n_threads_batch},");A(f"    n_ctx={B.n_ctx},");A(f"    n_batch={B.n_batch},");A(f"    n_gpu_layers={B.n_gpu_layers},");A(f"    use_mmap={B.use_mmap},");A(f"    use_mlock={B.use_mlock},");A(f"    verbose={B.verbose},");A(')');A('```');A('');A('## 🧠 Reasoning');A('');G={_L:'Parameter count',_M:'Layer count',_G:'`n_threads`',_H:'`n_threads_batch`',_I:'`n_ctx`',_C:'`n_gpu_layers`',_J:'`n_batch`',_K:'`use_mmap`',_D:'`use_mlock`'}
	for(F,H)in G.items():
		if F in B.reasoning:A(f"**{H}**  ");A(f"{B.reasoning[F]}");A('')
	return _N.join(D)
def ask_yes_no(prompt):
	try:return input(prompt+' [y/N] ').strip().lower()in('y','yes')
	except(EOFError,KeyboardInterrupt):return _F
def main():
	M='utf-8';F='store_true';B=argparse.ArgumentParser(description='Compute hardware-aware Llama() constructor parameters from model weight + quant.');G=B.add_mutually_exclusive_group(required=_B);G.add_argument('--weight-gb',type=float,help='GGUF file size in GB');G.add_argument('--params-b',type=float,help='Model parameter count in billions');B.add_argument('--quant',required=_B,help=f"Quant type, one of: {_E.join(sorted(QUANT_BPW))}");B.add_argument('--model-path',default=_Q,help='Path to pass as model_path');B.add_argument('--context',type=int,default=4096,help='Desired n_ctx');B.add_argument('--layers',type=int,default=_A,help='Exact total layer count (improves accuracy)');B.add_argument('--sample-seconds',type=float,default=0,help='Average available RAM over N seconds (10-30 recommended) instead of one snapshot');B.add_argument('--verbose-llama',action=F,help='Set verbose=True in the Llama() call');B.add_argument('--json-out',default='llama_params.json');B.add_argument('--md-out',default='llama_params_report.md');B.add_argument('--no-interactive',action=F);B.add_argument('--write-md',action=F);A=B.parse_args();C=A.quant.upper()
	if C not in QUANT_BPW:sys.exit(f"Unknown quant '{A.quant}'. Known: {_E.join(sorted(QUANT_BPW))}")
	if A.sample_seconds>0:N=max(10,min(30,A.sample_seconds));print(f"📊 Sampling available RAM over {N:.0f}s for a stable reading…")
	else:print('🔍 Detecting host (instant snapshot — pass --sample-seconds 10-30 to average over time)…')
	try:D=recommend(weight_gb=A.weight_gb,params_b=A.params_b,quant=C,model_path=A.model_path,context=A.context,total_layers=A.layers,sample_seconds=A.sample_seconds,verbose_llama=A.verbose_llama)
	except ValueError as O:sys.exit(str(O))
	H,I,J=D.host,D.params,D.weight_gb;print_report(H,I,J,C);K=Path(A.json_out);K.write_text(json.dumps(D.to_dict(),indent=2),encoding=M);print(f"✅  JSON → {K}");E=A.write_md
	if not E and not A.no_interactive:E=ask_yes_no('\n📄 Save a human-readable Markdown report?')
	if E:L=Path(A.md_out);L.write_text(generate_markdown(H,I,J,C),encoding=M);print(f"✅  Markdown → {L}")
if __name__=='__main__':main()