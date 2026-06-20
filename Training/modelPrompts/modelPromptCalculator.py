import math
from datetime import datetime


# =========================================================
# 1. QUANTIZATION TAXONOMY (CURRENT REALITY SNAPSHOT)
# =========================================================

QUANTIZATION = {
    "PRECISION_CORE": {
        "FP32": "stable",
        "FP16": "stable",
        "BF16": "stable",
        "FP8 (E4M3)": "stable",
        "FP8 (E5M2)": "stable",
    },

    "INTEGER_QUANT": {
        "INT8": "stable",
        "W8A8": "stable",
        "INT6": "emerging",
        "INT5": "emerging",
        "INT4": "stable",
        "INT3": "experimental",
        "INT2": "research",
    },

    "WEIGHT_ONLY": {
        "GPTQ": "stable",
        "AWQ": "stable",
        "RTN": "baseline",
        "EXL2": "stable",
        "GGUF": "stable",
    },

    "MIXED_PRECISION": {
        "W4A16": "stable",
        "W4A8": "emerging",
        "SmoothQuant": "method",
    },

    "HARDWARE_SPECIFIC": {
        "NVFP4": "emerging",
        "MXFP4": "emerging",
    },

    "SPARSITY": {
        "2:4 INT8": "stable",
        "Sparse FP16": "emerging",
        "Sparse INT4": "experimental",
    },

    "RESEARCH": {
        "Ternary": "research",
        "Binary": "research",
        "LogQuant": "research",
        "LearnedQuant": "research",
    }
}


# =========================================================
# 2. MODEL CAPABILITY ENGINE
# =========================================================

def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def log_scale(x):
    return math.log1p(x)


def model_profile(size_b, tuned, reasoning_tier):
    s = log_scale(size_b)

    instruction = clamp(20 + s * 18 + (10 if tuned else -5))
    reasoning = clamp(10 + s * 22 + reasoning_tier * 15 + (8 if tuned else 0))

    return instruction, reasoning


def prompt_policy(inst, reason):
    if inst < 45:
        return "HIGH STRUCTURE", "Step-by-step + strict schema + examples"
    elif inst < 70:
        return "STRUCTURED", "Role + constraints + optional examples"
    return "LIGHT STRUCTURE", "Goal + constraints only"


# =========================================================
# 3. QUANTIZATION SCORING (STABILITY-AWARE)
# =========================================================

def stability_score(level):
    return {
        "stable": 100,
        "baseline": 85,
        "emerging": 70,
        "method": 75,
        "experimental": 50,
        "research": 35
    }.get(level, 60)


def quant_compatibility(model_b, stability):
    base = log_scale(model_b)
    return clamp(stability_score(stability) + base * 6)


# =========================================================
# 4. MARKDOWN GENERATION (CORE OUTPUT)
# =========================================================

def generate_markdown(snapshot):
    now = datetime.utcnow().isoformat()

    md = [
        "# LLM Quantization & Prompt Intelligence Registry",
        "",
        f"**Generated:** {now}",
        "",
        "## System Type",
        "- Snapshot-driven (externally provided state)",
        "- Canonical quantization taxonomy + extensible registry",
        "- No deployment logic or CLI assumptions",
        "",
        "---",
        ""
    ]

    for m in snapshot["models"]:
        inst, reason = model_profile(
            m["size_b"],
            m["instruct_tuned"],
            m["reasoning_tier"]
        )

        level, template = prompt_policy(inst, reason)

        md.append(f"## Model: {m['name']} ({m['size_b']}B)")
        md.append("")
        md.append("### Prompt Profile")
        md.append(f"- Instruction Stability: {inst:.2f}/100")
        md.append(f"- Reasoning Power: {reason:.2f}/100")
        md.append(f"- Prompt Strategy: **{level}**")
        md.append("")
        md.append("### Recommended Template")
        md.append(template)
        md.append("")
        md.append("### Quantization Compatibility")

        for group, items in QUANTIZATION.items():
            md.append(f"\n#### {group}")
            for q, status in items.items():
                score = quant_compatibility(m["size_b"], status)
                md.append(f"- {q}: {score:.1f}/100 ({status})")

        md.append("\n---\n")

    return "\n".join(md)


# =========================================================
# 5. FILE WRITER
# =========================================================

def save(md, filename="model_prompt_registry.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md)


# =========================================================
# 6. ENTRY POINT (SIMPLE FUNCTION ONLY)
# =========================================================

def generate(snapshot):
    md = generate_markdown(snapshot)
    save(md)

    print("[SAVED] model_prompt_registry.md")
    print(f"[MODELS] {len(snapshot['models'])}")
    print(f"[GROUPS] {len(QUANTIZATION)}")
    print("[STATUS] Snapshot registry generated")


# =========================================================
# 7. EXAMPLE SNAPSHOT
# =========================================================

if __name__ == "__main__":

    snapshot = {
        "models": [
            {"name": "TinyLM", "size_b": 0.5, "instruct_tuned": False, "reasoning_tier": 0},
            {"name": "BaseLM", "size_b": 7, "instruct_tuned": True, "reasoning_tier": 1},
            {"name": "UltraLM", "size_b": 70, "instruct_tuned": True, "reasoning_tier": 2},
        ]
    }

    generate(snapshot)