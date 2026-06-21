_A='nvidia-smi'
import shutil,subprocess
BASE_REQUIREMENTS=['numpy>=2.4.6','pywebview>=6.2.1','scikit-learn>=1.8.0','sentence-transformers>=5.5.1','torch>=2.12.0','transformers>=4.45.0','fastapi>=0.137.1','uvicorn[standdard]>=0.49.0','websockets>=10.4', 'invokeai-python>=0.6.2']
CUDA_WHEELS={'12.1':'cu121','12.2':'cu122','12.3':'cu123','12.4':'cu124','12.5':'cu125','13.0':'cu130','13.2':'cu132'}
def has_nvidia():return shutil.which(_A)is not None
def get_cuda_version():
	B='CUDA Version:'
	try:
		C=subprocess.check_output([_A],stderr=subprocess.DEVNULL,text=True)
		for A in C.splitlines():
			if B in A:return A.split(B)[1].split()[0]
	except Exception:pass
def has_rocm():return shutil.which('rocminfo')is not None or shutil.which('rocm-smi')is not None
def generate_requirements():
	B='llama-cpp-python>=0.3.30';A=[];A.extend(BASE_REQUIREMENTS);A.append('')
	if has_nvidia():
		D=get_cuda_version();C=CUDA_WHEELS.get(D);A.append('# NVIDIA GPU detected')
		if C:A.append(f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/{C}")
		else:A.append('# NVIDIA GPU detected but CUDA version is unknown.');A.append('# Install the matching llama-cpp-python CUDA wheel manually.')
		A.append(B)
	elif has_rocm():A.append('# AMD ROCm detected');A.append('# Install llama-cpp-python with:');A.append('--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/rocm');A.append('# or build from source using:');A.append('# CMAKE_ARGS="-DGGML_HIP=on" pip install llama-cpp-python');A.append(B)
	else:A.append('# CPU-only installation');A.append(B);A.append(' # CMAKE_ARGS="-DLLAMA_NATIVE=ON -DLLAMA_OPENMP=ON" pip install llama-cpp-python')
	with open('requirements.txt','w',encoding='utf-8')as E:E.write('\n'.join(A))
	print('requirements.txt generated successfully.')
if __name__=='__main__':generate_requirements()