#!/usr/bin/env python3
import argparse
import gzip
import json
import zlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

def compute_sum_logprob(model, tokenizer, text, device, max_length=2048):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    # shift for next‐token logprobs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()

def main():
    parser = argparse.ArgumentParser(
        description="Membership inference via Δ = logp / zlib_len"
    )
    parser.add_argument(
        "--data", type=str, default="Magazine_Subscriptions.json.gz",
        help="Path to your gzipped JSON dataset"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max number of reviews to process (0 = all)"
    )
    parser.add_argument(
        "--output", type=str, default="membership_results.csv",
        help="CSV file to write results to"
    )
    args = parser.parse_args()

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    target_name = "meta-llama/Llama-3.2-1B"
    ref_name    = "meta-llama/Llama-3.2-3B"

    print(f"Loading target model ({target_name})…")
    tgt_tok = AutoTokenizer.from_pretrained(target_name, use_fast=False)
    tgt_mdl = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=torch.float16
    ).to(device)

    print(f"Loading reference model ({ref_name})…")
    ref_tok = AutoTokenizer.from_pretrained(ref_name, use_fast=False)
    ref_mdl = AutoModelForCausalLM.from_pretrained(
        ref_name, torch_dtype=torch.float16
    ).to(device)

    # Read dataset
    print(f"Reading records from {args.data}…")
    records = []
    with gzip.open(args.data, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)
    if args.limit > 0 and args.limit < total:
        total = args.limit
        records = records[:args.limit]
    print(f"→ Will process {total} reviews")

    # Process with progress bar
    results = []
    for rec in tqdm(records, desc="Processing reviews", total=total):
        text = rec.get("reviewText", "").strip()
        if not text:
            continue

        # zlib‐compressed length
        zlen = len(zlib.compress(text.encode("utf-8")))

        # sum log‐prob under each model
        sum_lp_tgt = compute_sum_logprob(tgt_mdl, tgt_tok, text, device)
        sum_lp_ref = compute_sum_logprob(ref_mdl, ref_tok, text, device)

        # normalized Δ scores
        delta_tgt = sum_lp_tgt / zlen
        delta_ref = sum_lp_ref / zlen
        delta_diff = delta_tgt - delta_ref

        results.append({
            "reviewID":     rec.get("reviewID"),
            "zlib_len":     zlen,
            "sum_logp_tgt": sum_lp_tgt,
            "sum_logp_ref": sum_lp_ref,
            "delta_tgt":    delta_tgt,
            "delta_ref":    delta_ref,
            "delta_diff":   delta_diff,
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Done → wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
