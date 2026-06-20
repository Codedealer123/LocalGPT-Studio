import shutil
import subprocess

BASE_REQUIREMENTS = [
    "numpy==2.4.6",
    "pywebview==6.2.1",
    "scikit-learn==1.8.0",
    "sentence-transformers==5.5.1",
    "torch==2.12.0",
    "transformers==4.45.0",
    "fastapi==0.137.1",
    "uvicorn==0.49.0",
    "websockets==10.4",
]

CUDA_WHEELS = {
    "12.1": "cu121",
    "12.2": "cu122",
    "12.3": "cu123",
    "12.4": "cu124",
    "12.5": "cu125",
    "13.0": "cu130",
    "13.2": "cu132",
}


def has_nvidia():
    return shutil.which("nvidia-smi") is not None


def get_cuda_version():
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True
        )

        for line in output.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].split()[0]

    except Exception:
        pass

    return None


def has_rocm():
    return (
        shutil.which("rocminfo") is not None
        or shutil.which("rocm-smi") is not None
    )


def generate_requirements():
    lines = []

    lines.extend(BASE_REQUIREMENTS)
    lines.append("")

    if has_nvidia():
        cuda_version = get_cuda_version()
        wheel = CUDA_WHEELS.get(cuda_version)

        lines.append("# NVIDIA GPU detected")

        if wheel:
            lines.append(
                f"--extra-index-url "
                f"https://abetlen.github.io/llama-cpp-python/whl/{wheel}"
            )
        else:
            lines.append(
                "# NVIDIA GPU detected but CUDA version is unknown."
            )
            lines.append(
                "# Install the matching llama-cpp-python CUDA wheel manually."
            )

        lines.append("llama-cpp-python>=0.3.30")

    elif has_rocm():
        lines.append("# AMD ROCm detected")
        lines.append(
            "# Install llama-cpp-python with:"
        )
        lines.append(
            "--extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/rocm"
        )
        lines.append(
            "# or build from source using:"
        )
        lines.append(
            "# CMAKE_ARGS=\"-DGGML_HIP=on\" pip install llama-cpp-python"
        )
        lines.append("llama-cpp-python>=0.3.30")

    else:
        lines.append("# CPU-only installation")
        lines.append("llama-cpp-python>=0.3.30")
        lines.append(" # CMAKE_ARGS=\"-DLLAMA_NATIVE=ON -DLLAMA_OPENMP=ON\" pip install llama-cpp-python")

    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("requirements.txt generated successfully.")


if __name__ == "__main__":
    generate_requirements()