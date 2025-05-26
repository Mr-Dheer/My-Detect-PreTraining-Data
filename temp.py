import json, gzip, csv, zlib, argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_dataset(path):
    """
    Load a JSON Lines dataset from a path. Supports .gz compressed files.
    Each line should be a JSON object.
    """
    open_fn = gzip.open if path.endswith('.gz') else open
    data = []
    with open_fn(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return data

def compute_perplexity(model, tokenizer, text, stride=512, max_length=2048, device='cuda'):
    """
    Compute perplexity for a long text by splitting into overlapping chunks.
    """
    encodings = tokenizer(text, return_tensors='pt', truncation=False)
    input_ids = encodings.input_ids.to(device)
    nlls = []
    seq_len = input_ids.size(1)
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i
        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_slice, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / seq_len)
    return ppl.item()

def compute_zlib_ratio(text):
    """
    Compute compression ratio: compressed size divided by raw size.
    """
    raw = text.encode('utf-8')
    if not raw:
        return 0
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Membership inference for Llama pretraining data detection'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='/home/kavach/Dev/PythonProject/Magazine_Subscriptions.json.gz',
        help='Path to JSON Lines dataset (.gz OK)'
    )
    parser.add_argument(
        '--llama',
        type=str,
        default='meta-llama/Llama-3.2-3B',
        help='Target LLM model name'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='meta-llama/Llama-3.2-1B',
        help='Baseline LLM model name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Torch device (e.g., cuda or cpu)'
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading target model: {args.llama}")
    tok_target = AutoTokenizer.from_pretrained(args.llama)
    model_target = AutoModelForCausalLM.from_pretrained(
        args.llama,
        device_map='auto',
        torch_dtype=torch.float16
    )

    print(f"Loading baseline model: {args.baseline}")
    tok_base = AutoTokenizer.from_pretrained(args.baseline)
    model_base = AutoModelForCausalLM.from_pretrained(
        args.baseline,
        device_map='auto',
        torch_dtype=torch.float16
    )

    # Put both models in eval mode
    model_target.eval()
    model_base.eval()

    print(f"Loading dataset from {args.data}")
    data = load_dataset(args.data)
    print(f"Loaded {len(data)} records.")

    print("Processing samples...")
    with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=['id', 'ppl_target', 'ppl_base', 'ppl_ratio', 'zlib_ratio']
        )
        writer.writeheader()

        for idx, sample in enumerate(tqdm(data, desc='Evaluating samples')):
            text = sample.get('text') or sample.get('reviewText') or ''
            sid = sample.get('id') or sample.get('reviewerID') or f"sample_{idx}"
            if not text:
                continue
            ppl_t = compute_perplexity(
                model_target, tok_target, text, device=device
            )
            ppl_b = compute_perplexity(
                model_base, tok_base, text, device=device
            )
            ratio = ppl_b / ppl_t if ppl_t > 0 else 0
            zratio = compute_zlib_ratio(text)
            writer.writerow({
                'id': sid,
                'ppl_target': ppl_t,
                'ppl_base': ppl_b,
                'ppl_ratio': ratio,
                'zlib_ratio': zratio
            })

    print(f"Done. Results written to {args.output}")
