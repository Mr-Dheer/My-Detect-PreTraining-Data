#!/usr/bin/env python3
import argparse
import zlib
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random

def compute_perplexity(model, tokenizer, text, device,
                       max_length=2048, stride=512):
    """Sliding-window PPL like HF examples."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n_tokens = enc.size(0)
    if n_tokens == 0:
        return float("nan")
    nlls, count = [], 0
    for i in range(0, n_tokens, stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, n_tokens)
        inp = enc[begin:end].unsqueeze(0).to(device)
        tgt = inp.clone()
        # only predict the last (end - i) tokens
        tgt[:, : (end - begin) - (end - i)] = -100
        with torch.no_grad():
            out = model(inp, labels=tgt)
        # out.loss is avg NLL over predicted tokens
        window_nll = out.loss * (end - i)
        nlls.append(window_nll)
        count += (end - i)
    avg_nll = torch.stack(nlls).sum() / count
    return torch.exp(avg_nll).item()

def main():
    p = argparse.ArgumentParser("Zlib-based Extraction Pipeline")
    p.add_argument("--model", type=str,
                   default="meta-llama/Llama-3.2-1B",
                   help="HF model to sample & score")
    p.add_argument("--num", type=int, default=89689,
                   help="Total number of candidates to generate")
    p.add_argument("--batch", type=int, default=16,
                   help="Generate this many samples per model.generate call")
    p.add_argument("--gen_len", type=int, default=64,
                   help="New-token length for each sample")
    p.add_argument("--temp", type=float, default=1.0,
                   help="Sampling temperature")
    p.add_argument("--top_k", type=int, default=50,
                   help="Top-K sampling")
    p.add_argument("--top_p", type=float, default=0.95,
                   help="Top-P (nucleus) sampling")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--output", type=str,
                   default="extraction_zlib.csv",
                   help="CSV to write: text,ppl,zlib_len,ratio")
    p.add_argument("--keep", type=int, default=100,
                   help="How many top candidates to keep in the CSV")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load model & tokenizer
    print(f"Loading {args.model} …")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)
    mdl.eval()

    # 2) generate candidates
    candidates = []
    print(f"Generating {args.num} samples…")
    while len(candidates) < args.num:
        # empty prompt → just generate from BOS token
        inp = torch.tensor([[tok.bos_token_id]],
                           device=device)
        out = mdl.generate(
            inp,
            max_new_tokens=args.gen_len,
            do_sample=True,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            num_return_sequences=min(args.batch, args.num - len(candidates))
        )
        # decode only the new tokens
        for seq in out:
            gen = seq[1:].tolist()  # drop BOS
            text = tok.decode(gen, skip_special_tokens=True)
            candidates.append(text)
        # progress
        print(f" → {len(candidates)}/{args.num}", end="\r")

    # 3) score each candidate
    print("\nScoring candidates…")
    rows = []
    for text in tqdm(candidates, desc="Scoring"):
        # a) PPL
        ppl = compute_perplexity(mdl, tok, text, device)
        # b) zlib length
        zlen = len(zlib.compress(text.encode("utf-8")))
        # c) ratio: lower ⇒ more suspicious
        ratio = ppl / zlen if zlen > 0 else float("inf")
        rows.append((text, ppl, zlen, ratio))

    # 4) sort & keep top-K
    rows.sort(key=lambda x: x[3])
    to_write = rows[: args.keep]

    # 5) dump CSV
    print(f"Writing top {len(to_write)} to {args.output}")
    with open(args.output, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["text", "ppl", "zlib_len", "ppl/zlib"])
        for text, ppl, zlen, ratio in to_write:
            writer.writerow([text, ppl, zlen, ratio])

    print("Done.")

if __name__ == "__main__":
    main()
