import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = "./intent_classifier"

# ------------------------------------------------------------------
# Load label mapping
# ------------------------------------------------------------------

with open(f"{MODEL_DIR}/label_map.json", "r") as f:
    label_map = json.load(f)

ID2LABEL = {int(k): v for k, v in label_map["id2label"].items()}
LABEL2ID = {v: int(k) for k, v in ID2LABEL.items()}

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

torch.set_grad_enabled(False)

if device == "cuda":
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def predict_intent(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()

    return {
        "intent": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id]),
    }


# ------------------------------------------------------------------
# Benchmark (speed only)
# ------------------------------------------------------------------

def benchmark(text="Generate an image of a futuristic city at night", runs=1000):

    for _ in range(20):
        predict_intent(text)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        predict_intent(text)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / runs) * 1000
    qps = runs / elapsed

    print("\nSpeed Benchmark")
    print("-" * 40)
    print(f"Device       : {device}")
    print(f"Runs         : {runs}")
    print(f"Average      : {avg_ms:.3f} ms")
    print(f"Throughput   : {qps:.2f} req/sec")
    print("-" * 40)


# ------------------------------------------------------------------
# Evaluation (accuracy + optional F1)
# ------------------------------------------------------------------

def load_dataset(path="./test.json"):
    with open(path, "r") as f:
        return json.load(f)


def evaluate(path="./test.json", use_f1=True):

    data = load_dataset(path)

    y_true = []
    y_pred = []

    # warmup
    for _ in range(20):
        predict_intent("warmup text")

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for item in data:
        pred = predict_intent(item["text"])
        y_pred.append(pred["intent"])
        y_true.append(item["label"])

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_true, y_pred)

    print("\nEvaluation Results")
    print("-" * 40)
    print(f"Samples      : {len(data)}")
    print(f"Accuracy     : {acc:.4f}")

    if use_f1:
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"F1 Macro     : {f1:.4f}")

    avg_ms = (elapsed / len(data)) * 1000
    qps = len(data) / elapsed

    print(f"Avg latency  : {avg_ms:.3f} ms")
    print(f"Throughput   : {qps:.2f} req/sec")
    print("-" * 40)


# ------------------------------------------------------------------
# Interactive
# ------------------------------------------------------------------

def main():
    print(f"Intent Classifier Ready ({device})")
    print("Commands:")
    print("  benchmark")
    print("  eval")
    print("  exit")
    print()

    while True:
        text = input("> ").strip()

        if not text:
            continue

        if text.lower() in {"exit", "quit"}:
            break

        if text.lower() == "benchmark":
            benchmark()
            continue

        if text.lower() == "eval":
            evaluate("./test.json")
            continue

        start = time.perf_counter()

        result = predict_intent(text)

        if device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        print()
        print(f"Intent     : {result['intent']}")
        print(f"Confidence : {result['confidence']:.4f}")
        print(f"Latency    : {latency_ms:.2f} ms")
        print()


if __name__ == "__main__":
    main()