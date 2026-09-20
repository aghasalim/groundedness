# -*- coding: utf-8 -*-
"""Inference for the trained detector: answer + sources -> unsupported spans.
Shared by the benchmark eval and the server. CPU by default, int8 dynamic
quantisation optional (the Pi)."""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class Detector:
    def __init__(self, path, quantize=False, threshold=0.5, max_len=512):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).eval()
        if quantize:
            # int8 linears halve the CPU time; needs a quantised engine (qnnpack on arm, fbgemm on x86)
            try:
                if 'qnnpack' in torch.backends.quantized.supported_engines: torch.backends.quantized.engine = 'qnnpack'
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception as e:
                print('quantisation unavailable, fp32:', e)
        self.threshold, self.max_len = threshold, max_len
        # the trainer writes its dev-calibrated threshold next to the weights
        try:
            import json, os
            t = json.load(open(os.path.join(path, 'training.json'))).get('threshold')
            if t and threshold == 0.5: self.threshold = float(t)
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self, answer, sources):
        src = sources if isinstance(sources, str) else '\n\n'.join(sources)
        e = self.tok(answer, src, truncation='longest_first', max_length=self.max_len, return_offsets_mapping=True, return_tensors='pt')
        off = e.pop('offset_mapping')[0].tolist(); seq = e.sequence_ids()
        prob = self.model(**e).logits.softmax(-1)[0, :, 1].tolist()
        spans, cur = [], None
        for i, (s, t) in enumerate(off):
            if seq[i] != 0 or t == s: continue
            if prob[i] >= self.threshold:
                if cur and s - cur[1] <= 1: cur[1] = t
                else:
                    if cur: spans.append(cur)
                    cur = [s, t]
            elif cur: spans.append(cur); cur = None
        if cur: spans.append(cur)
        # drop one-character fragments (punctuation the tokenizer split off)
        spans = [[s, t] for s, t in spans if len(answer[s:t].strip()) > 1]
        bad = sum(t - s for s, t in spans)
        return {'spans': spans, 'unsupported': [answer[s:t].strip() for s, t in spans], 'score': max(0.0, 1 - bad / max(1, len(answer))), 'grounded': not spans}

if __name__ == '__main__':
    import sys, json
    d = Detector(sys.argv[1])
    print(json.dumps(d(sys.argv[2], sys.argv[3:]), ensure_ascii=False, indent=1))
