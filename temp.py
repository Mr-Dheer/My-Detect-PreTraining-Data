import os
import math
import zlib
import random
import json
import gzip
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

# ======================================
# Configuration and Model Loading
# ======================================

# Target model for extraction attack
TARGET_MODEL_NAME = 'meta-llama/Llama-3.2-1B'
# Reference model for membership inference ratios
REF_MODEL_NAME = 'meta-llama/Llama-3.2-3B'


# Quantization for reduced memory
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

#
# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
ref_tokenizer = AutoTokenizer.from_pretrained(REF_MODEL_NAME)

# Load models with 8-bit quantization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL_NAME,
    device_map='auto',
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True
)
model.eval()


ref_model = AutoModelForCausalLM.from_pretrained(
    REF_MODEL_NAME,
    device_map='auto',
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    # low_cpu_mem_usage=True
)
ref_model.eval()

# ======================================
# Sampling Strategies
# ======================================

def sample_with_temperature(prefix, temperature=1.0, top_k=50, top_p=0.95, max_new_tokens=50):
    inputs = tokenizer(prefix, return_tensors='pt').to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    # Return only the newly generated tokens
    gen_ids = output_ids[0, inputs.input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

# Decaying temperature schedule
def sample_with_decay_temperature(prefix, initial_temp=1.0, decay=0.95, steps=10, **kwargs):
    text = ''
    temp = initial_temp
    for _ in range(steps):
        token = sample_with_temperature(prefix + text, temperature=temp, **kwargs)
        text += token
        temp *= decay
    return text

# Internet-prefix strategy
INTERNET_PREFIXES = [
    "Breaking News:\n", "In a recent report:\n", "According to sources:\n"
]

def sample_with_internet_prefix(prefix, **kwargs):
    iprefix = random.choice(INTERNET_PREFIXES)
    return sample_with_temperature(iprefix + prefix, **kwargs)

# ======================================
# Membership Inference Signals
# ======================================

def compute_perplexity(prefix, continuation, model_obj, tok):
    text = prefix + continuation
    inputs = tok(text, return_tensors='pt').to(model_obj.device)
    with torch.no_grad():
        outputs = model_obj(**inputs, labels=inputs.input_ids)
        ppl = torch.exp(outputs.loss)
    return ppl.item()

# Sliding window perplexity
def compute_sliding_ppl(text, model_obj, tok, window_size=50):
    tokens = tok(text, return_tensors='pt').input_ids[0]
    min_ppl = float('inf')
    for i in range(0, tokens.size(0) - window_size + 1):
        window = tok.decode(tokens[i:i+window_size])
        ppl = compute_perplexity('', window, model_obj, tok)
        min_ppl = min(min_ppl, ppl)
    return min_ppl

# Lowercase perplexity ratio
def compute_lowercase_ratio(prefix, continuation, model_obj, tok):
    orig = compute_perplexity(prefix, continuation, model_obj, tok)
    low = compute_perplexity(prefix.lower(), continuation.lower(), model_obj, tok)
    return orig / low if low > 0 else float('inf')

# Zlib compression ratio
def compute_zlib_ratio(text):
    raw = text.encode()
    comp = zlib.compress(raw)
    return len(comp) / len(raw) if len(raw) > 0 else 1.0

# ======================================
# Data Loading
# ======================================

def load_reviews(path, max_records=None):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_records and idx >= max_records:
                break
            yield json.loads(line)

# ======================================
# Deduplication (trigram Jaccard)
# ======================================

def jaccard_trigrams(a, b):
    def trigrams(s):
        return set([s[i:i+3] for i in range(len(s)-2)])
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0

# ======================================
# Main Attack Loop
# ======================================

def main(args):
    records = list(load_reviews(args.input, max_records=args.max_records))
    results = []

    for entry in tqdm(records, desc='Processing Reviews'):
        prefix = entry.get('reviewText', '')
        # Generate candidates
        cand1 = sample_with_temperature(prefix, temperature=1.0, max_new_tokens=args.max_new_tokens)
        cand2 = sample_with_decay_temperature(prefix, steps=args.decay_steps, max_new_tokens=args.max_new_tokens)
        cand3 = sample_with_internet_prefix(prefix, max_new_tokens=args.max_new_tokens)
        for sample in [cand1, cand2, cand3]:
            # Signals
            ppl = compute_perplexity(prefix, sample, model, tokenizer)
            ppl_ref = compute_perplexity(prefix, sample, ref_model, ref_tokenizer)
            slide_ppl = compute_sliding_ppl(prefix + sample, model, tokenizer)
            low_ratio = compute_lowercase_ratio(prefix, sample, model, tokenizer)
            z_ratio = compute_zlib_ratio(prefix + sample)

            results.append({
                'entry_id': entry.get('reviewerID'),
                'sample': sample,
                'ppl': ppl,
                'ppl_ratio_ref': ppl / ppl_ref if ppl_ref>0 else float('inf'),
                'slide_ppl': slide_ppl,
                'lowercase_ratio': low_ratio,
                'zlib_ratio': z_ratio
            })

    # Deduplicate: keep unique by trigram Jaccard < threshold
    unique_results = []
    for res in results:
        if all(jaccard_trigrams(res['sample'], u['sample']) < args.dedup_threshold for u in unique_results):
            unique_results.append(res)

    # Rank by desired metric
    # topk = sorted(unique_results, key=lambda x: x[args.metric], reverse=True)[:args.top_k]
    topk = sorted(unique_results, key=lambda x: x[args.metric], reverse=True)[:args.top_k]

    # Output results
    with open(args.output, 'w') as out_f:
        json.dump(topk, out_f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='Magazine_Subscriptions.json.gz')
    parser.add_argument('--output', type=str, default='attack_results.json')
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--metric', type=str, default='ppl_ratio_ref',
                        choices=['ppl', 'ppl_ratio_ref', 'slide_ppl', 'lowercase_ratio', 'zlib_ratio'])
    parser.add_argument('--dedup_threshold', type=float, default=0.8)
    parser.add_argument('--decay_steps', type=int, default=20)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    args = parser.parse_args()
    main(args)
