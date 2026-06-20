# LLM Quantization & Prompt Intelligence Registry

**Generated:** 2026-06-14T09:44:00.733508

## System Type
- Snapshot-driven (externally provided state)
- Canonical quantization taxonomy + extensible registry
- No deployment logic or CLI assumptions

---

## Model: TinyLM (0.5B)

### Prompt Profile
- Instruction Stability: 22.30/100
- Reasoning Power: 18.92/100
- Prompt Strategy: **HIGH STRUCTURE**

### Recommended Template
Step-by-step + strict schema + examples

### Quantization Compatibility

#### PRECISION_CORE
- FP32: 100.0/100 (stable)
- FP16: 100.0/100 (stable)
- BF16: 100.0/100 (stable)
- FP8 (E4M3): 100.0/100 (stable)
- FP8 (E5M2): 100.0/100 (stable)

#### INTEGER_QUANT
- INT8: 100.0/100 (stable)
- W8A8: 100.0/100 (stable)
- INT6: 72.4/100 (emerging)
- INT5: 72.4/100 (emerging)
- INT4: 100.0/100 (stable)
- INT3: 52.4/100 (experimental)
- INT2: 37.4/100 (research)

#### WEIGHT_ONLY
- GPTQ: 100.0/100 (stable)
- AWQ: 100.0/100 (stable)
- RTN: 87.4/100 (baseline)
- EXL2: 100.0/100 (stable)
- GGUF: 100.0/100 (stable)

#### MIXED_PRECISION
- W4A16: 100.0/100 (stable)
- W4A8: 72.4/100 (emerging)
- SmoothQuant: 77.4/100 (method)

#### HARDWARE_SPECIFIC
- NVFP4: 72.4/100 (emerging)
- MXFP4: 72.4/100 (emerging)

#### SPARSITY
- 2:4 INT8: 100.0/100 (stable)
- Sparse FP16: 72.4/100 (emerging)
- Sparse INT4: 52.4/100 (experimental)

#### RESEARCH
- Ternary: 37.4/100 (research)
- Binary: 37.4/100 (research)
- LogQuant: 37.4/100 (research)
- LearnedQuant: 37.4/100 (research)

---

## Model: BaseLM (7B)

### Prompt Profile
- Instruction Stability: 67.43/100
- Reasoning Power: 78.75/100
- Prompt Strategy: **STRUCTURED**

### Recommended Template
Role + constraints + optional examples

### Quantization Compatibility

#### PRECISION_CORE
- FP32: 100.0/100 (stable)
- FP16: 100.0/100 (stable)
- BF16: 100.0/100 (stable)
- FP8 (E4M3): 100.0/100 (stable)
- FP8 (E5M2): 100.0/100 (stable)

#### INTEGER_QUANT
- INT8: 100.0/100 (stable)
- W8A8: 100.0/100 (stable)
- INT6: 82.5/100 (emerging)
- INT5: 82.5/100 (emerging)
- INT4: 100.0/100 (stable)
- INT3: 62.5/100 (experimental)
- INT2: 47.5/100 (research)

#### WEIGHT_ONLY
- GPTQ: 100.0/100 (stable)
- AWQ: 100.0/100 (stable)
- RTN: 97.5/100 (baseline)
- EXL2: 100.0/100 (stable)
- GGUF: 100.0/100 (stable)

#### MIXED_PRECISION
- W4A16: 100.0/100 (stable)
- W4A8: 82.5/100 (emerging)
- SmoothQuant: 87.5/100 (method)

#### HARDWARE_SPECIFIC
- NVFP4: 82.5/100 (emerging)
- MXFP4: 82.5/100 (emerging)

#### SPARSITY
- 2:4 INT8: 100.0/100 (stable)
- Sparse FP16: 82.5/100 (emerging)
- Sparse INT4: 62.5/100 (experimental)

#### RESEARCH
- Ternary: 47.5/100 (research)
- Binary: 47.5/100 (research)
- LogQuant: 47.5/100 (research)
- LearnedQuant: 47.5/100 (research)

---

## Model: UltraLM (70B)

### Prompt Profile
- Instruction Stability: 100.00/100
- Reasoning Power: 100.00/100
- Prompt Strategy: **LIGHT STRUCTURE**

### Recommended Template
Goal + constraints only

### Quantization Compatibility

#### PRECISION_CORE
- FP32: 100.0/100 (stable)
- FP16: 100.0/100 (stable)
- BF16: 100.0/100 (stable)
- FP8 (E4M3): 100.0/100 (stable)
- FP8 (E5M2): 100.0/100 (stable)

#### INTEGER_QUANT
- INT8: 100.0/100 (stable)
- W8A8: 100.0/100 (stable)
- INT6: 95.6/100 (emerging)
- INT5: 95.6/100 (emerging)
- INT4: 100.0/100 (stable)
- INT3: 75.6/100 (experimental)
- INT2: 60.6/100 (research)

#### WEIGHT_ONLY
- GPTQ: 100.0/100 (stable)
- AWQ: 100.0/100 (stable)
- RTN: 100.0/100 (baseline)
- EXL2: 100.0/100 (stable)
- GGUF: 100.0/100 (stable)

#### MIXED_PRECISION
- W4A16: 100.0/100 (stable)
- W4A8: 95.6/100 (emerging)
- SmoothQuant: 100.0/100 (method)

#### HARDWARE_SPECIFIC
- NVFP4: 95.6/100 (emerging)
- MXFP4: 95.6/100 (emerging)

#### SPARSITY
- 2:4 INT8: 100.0/100 (stable)
- Sparse FP16: 95.6/100 (emerging)
- Sparse INT4: 75.6/100 (experimental)

#### RESEARCH
- Ternary: 60.6/100 (research)
- Binary: 60.6/100 (research)
- LogQuant: 60.6/100 (research)
- LearnedQuant: 60.6/100 (research)

---
