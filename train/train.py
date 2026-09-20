# -*- coding: utf-8 -*-
"""Token classifier: which words of an answer are not supported by its sources.

    python train/train.py --base xlm-roberta-base --out train/out/grounded-multilingual-base --epochs 3

Input is  <s> answer </s></s> sources </s>  so the answer is never truncated;
labels: 1 = inside an unsupported span, 0 = supported, -100 elsewhere.
"""
import argparse, json, os, random, time, math
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))

def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]

def encode(tok, rows, max_len):
    out = []
    for r in rows:
        e = tok(r['answer'], r['source'], truncation='longest_first', max_length=max_len, return_offsets_mapping=True)
        seq = e.sequence_ids(); labels = []
        for i, (s, t) in enumerate(e['offset_mapping']):
            if seq[i] != 0 or t == s: labels.append(-100); continue
            labels.append(1 if any(a < t and s < b for a, b in r['spans']) else 0)
        out.append({'input_ids': e['input_ids'], 'attention_mask': e['attention_mask'], 'labels': labels})
    return out

def batches(items, bs, pad, shuffle, rnd):
    idx = sorted(range(len(items)), key=lambda i: len(items[i]['input_ids']))
    chunks = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    if shuffle: rnd.shuffle(chunks)
    for ch in chunks:
        L = max(len(items[i]['input_ids']) for i in ch)
        ids = torch.full((len(ch), L), pad); am = torch.zeros((len(ch), L), dtype=torch.long); lab = torch.full((len(ch), L), -100)
        for j, i in enumerate(ch):
            n = len(items[i]['input_ids']); ids[j, :n] = torch.tensor(items[i]['input_ids']); am[j, :n] = 1; lab[j, :n] = torch.tensor(items[i]['labels'])
        yield ids, am, lab

@torch.no_grad()
def evaluate(model, items, dev, bs, pad):
    model.eval(); tp = fp = fn = 0; ex_tp = ex_fp = ex_fn = ex_tn = 0
    for ids, am, lab in batches(items, bs, pad, False, None):
        logits = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
        pred = logits.argmax(-1).cpu()
        m = lab != -100
        tp += int(((pred == 1) & (lab == 1) & m).sum()); fp += int(((pred == 1) & (lab == 0) & m).sum()); fn += int(((pred == 0) & (lab == 1) & m).sum())
        for j in range(ids.shape[0]):
            g = bool(((lab[j] == 1)).any()); p = bool(((pred[j] == 1) & m[j]).any())
            ex_tp += g and p; ex_fp += (not g) and p; ex_fn += g and not p; ex_tn += (not g) and not p
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F = 2 * P * R / max(1e-9, P + R)
    model.train()
    return {'token_p': round(P, 3), 'token_r': round(R, 3), 'token_f1': round(F, 3), 'answer_recall': round(ex_tp / max(1, ex_tp + ex_fn), 3), 'answer_false_alarm': round(ex_fp / max(1, ex_fp + ex_tn), 3)}

@torch.no_grad()
def calibrate(model, items, dev, bs, pad):
    model.eval(); probs, gold = [], []
    for ids, am, lab in batches(items, bs, pad, False, None):
        p = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits.softmax(-1)[..., 1].cpu()
        for j in range(ids.shape[0]):
            m = lab[j] != -100
            probs.append(p[j][m]); gold.append(bool((lab[j] == 1).any()))
    best = (0.5, -1.0)
    for t in [x / 100 for x in range(30, 96, 2)]:
        flag = [bool((pr >= t).any()) for pr in probs]
        rec = sum(f and g for f, g in zip(flag, gold)) / max(1, sum(gold))
        fa = sum(f and not g for f, g in zip(flag, gold)) / max(1, sum(not g for g in gold))
        score = rec if fa <= 0.10 else rec - (fa - 0.10) * 3
        if score > best[1]: best = (t, score)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='xlm-roberta-base'); ap.add_argument('--out', default=os.path.join(HERE, 'out', 'grounded-multilingual-base'))
    ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--max-len', type=int, default=384); ap.add_argument('--pos-weight', type=float, default=3.0); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    rnd = random.Random(0); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.base)
    train_rows = load(os.path.join(HERE, 'data', 'train.jsonl')); dev_rows = load(os.path.join(HERE, 'data', 'dev.jsonl'))
    if a.limit: train_rows = train_rows[:a.limit]; dev_rows = dev_rows[:max(50, a.limit // 10)]
    tr = encode(tok, train_rows, a.max_len); dv = encode(tok, dev_rows, a.max_len)
    print(f'device {dev} · train {len(tr)} · dev {len(dv)} · base {a.base}', flush=True)
    model = AutoModelForTokenClassification.from_pretrained(a.base, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.bs); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, a.pos_weight]).to(dev), ignore_index=-100)
    best, t0, step = -1, time.time(), 0
    for ep in range(a.epochs):
        for ids, am, lab in batches(tr, a.bs, tok.pad_token_id, True, rnd):
            logits = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
            loss = lossf(logits.view(-1, 2), lab.to(dev).view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            # variable-length batches fragment the MPS cache until the box swaps; hand memory back often
            if dev.type == 'mps' and step % 2 == 0: torch.mps.empty_cache()
            if step % 50 == 0: print(f'ep {ep} step {step}/{steps} loss {loss.item():.4f} {int(time.time()-t0)}s', flush=True)
        if dev.type == 'mps': torch.mps.empty_cache()
        m = evaluate(model, dv, dev, a.bs, tok.pad_token_id); print(f'epoch {ep} dev {m}', flush=True)
        if dev.type == 'mps': torch.mps.empty_cache()
        if m['token_f1'] > best:
            best = m['token_f1']; model.save_pretrained(a.out); tok.save_pretrained(a.out)
            json.dump({'base': a.base, 'epoch': ep, 'dev': m, 'train_answers': len(tr), 'pos_weight': a.pos_weight, 'lr': a.lr, 'name': 'siba/grounded-multilingual-base'}, open(os.path.join(a.out, 'training.json'), 'w'), indent=1)
            print('saved', a.out, flush=True)
    # Pick the threshold on dev: the highest answer-level recall whose false-alarm
    # rate stays under 10 %, written next to the weights for detect.py to read.
    model = AutoModelForTokenClassification.from_pretrained(a.out).to(dev)
    thr = calibrate(model, dv, dev, a.bs, tok.pad_token_id)
    info = json.load(open(os.path.join(a.out, 'training.json'))); info['threshold'] = thr
    json.dump(info, open(os.path.join(a.out, 'training.json'), 'w'), indent=1)
    print('done best token_f1', best, 'threshold', thr, flush=True)

if __name__ == '__main__':
    main()
