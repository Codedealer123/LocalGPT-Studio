_E = 'confidence'
_D = './test.json'
_C = 'intent'
_B = True
_A = 'cuda'
import json, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
MODEL_DIR = './intent_classifier'
with open(f'{MODEL_DIR}/label_map.json', 'r') as f:
    label_map = json.load(f)
ID2LABEL = {int(A): B for A, B in label_map['id2label'].items()}
LABEL2ID = {B: int(A) for A, B in ID2LABEL.items()}
device = _A if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
torch.set_grad_enabled(False)
if device == _A:
    torch.backends.cudnn.benchmark = _B

def predict_intent(text):
    A = tokenizer(text, return_tensors='pt', truncation=_B, max_length=128)
    A = {A: B.to(device) for A, B in A.items()}
    with torch.inference_mode():
        D = model(**A).logits
    B = torch.softmax(D, dim=-1)[0]
    C = torch.argmax(B).item()
    return {_C: ID2LABEL[C], _E: float(B[C])}

def benchmark(text='Generate an image of a futuristic city at night', runs=1000):
    A = runs
    for C in range(20):
        predict_intent(text)
    if device == _A:
        torch.cuda.synchronize()
    D = time.perf_counter()
    for C in range(A):
        predict_intent(text)
    if device == _A:
        torch.cuda.synchronize()
    B = time.perf_counter() - D
    E = B / A * 1000
    F = A / B
    print('\nSpeed Benchmark')
    print('-' * 40)
    print(f'Device       : {device}')
    print(f'Runs         : {A}')
    print(f'Average      : {E:.3f} ms')
    print(f'Throughput   : {F:.2f} req/sec')
    print('-' * 40)

def load_dataset(path=_D):
    with open(path, 'r') as A:
        return json.load(A)

def evaluate(path=_D, use_f1=_B):
    A = load_dataset(path)
    B = []
    C = []
    for L in range(20):
        predict_intent('warmup text')
    if device == _A:
        torch.cuda.synchronize()
    F = time.perf_counter()
    for D in A:
        G = predict_intent(D['text'])
        C.append(G[_C])
        B.append(D['label'])
    if device == _A:
        torch.cuda.synchronize()
    E = time.perf_counter() - F
    H = accuracy_score(B, C)
    print('\nEvaluation Results')
    print('-' * 40)
    print(f'Samples      : {len(A)}')
    print(f'Accuracy     : {H:.4f}')
    if use_f1:
        I = f1_score(B, C, average='macro')
        print(f'F1 Macro     : {I:.4f}')
    J = E / len(A) * 1000
    K = len(A) / E
    print(f'Avg latency  : {J:.3f} ms')
    print(f'Throughput   : {K:.2f} req/sec')
    print('-' * 40)

def main():
    print(f'Intent Classifier Ready ({device})')
    print('Commands:')
    print('  benchmark')
    print('  eval')
    print('  exit')
    print()
    while _B:
        A = input('> ').strip()
        if not A:
            continue
        if A.lower() in {'exit', 'quit'}:
            break
        if A.lower() == 'benchmark':
            benchmark()
            continue
        if A.lower() == 'eval':
            evaluate(_D)
            continue
        C = time.perf_counter()
        B = predict_intent(A)
        if device == _A:
            torch.cuda.synchronize()
        D = (time.perf_counter() - C) * 1000
        print()
        print(f'Intent     : {B[_C]}')
        print(f'Confidence : {B[_E]:.4f}')
        print(f'Latency    : {D:.2f} ms')
        print()
if __name__ == '__main__':
    main()
