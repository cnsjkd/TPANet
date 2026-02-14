"""
Precompute cached prompt token embeddings (BERT last_hidden_state) for SEED chunk files.

This writes FP16 caches under: prompt_cache/*.pt

Typical training (cached):
  PROMPT_EMB_MODE=cached PROMPT_MODE=original python src/seed_emotion/model_training.py
"""
import os
import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import BertModel, BertTokenizer


def _is_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        if file_path.stat().st_size > 2048:
            return False
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1")


def _validate_local_bert_weights(bert_path: str) -> None:
    model_dir = Path(bert_path)
    if not model_dir.is_dir():
        return
    pytorch_bin = model_dir / "pytorch_model.bin"
    safetensors_file = model_dir / "model.safetensors"
    if _is_git_lfs_pointer(pytorch_bin):
        if safetensors_file.is_file() and safetensors_file.stat().st_size > 1024 * 1024:
            return
        raise RuntimeError(
            f"检测到无效模型权重: {pytorch_bin}\n"
            "该文件是 Git LFS 指针，不是真实的 BERT 权重。\n"
            "请下载完整权重到该目录（任选其一）：\n"
            "  - pytorch_model.bin\n"
            "  - model.safetensors\n"
            "下载完成后重新运行脚本即可。"
        )

def generate_statistics(batch_data):
    min_values = torch.min(batch_data.view(batch_data.size(0), -1), dim=1).values
    max_values = torch.max(batch_data.view(batch_data.size(0), -1), dim=1).values
    median_values = torch.median(batch_data.view(batch_data.size(0), -1), dim=1).values
    trends = torch.mean(batch_data[:, :, 1:] - batch_data[:, :, :-1], dim=(1, 2))
    return min_values, max_values, median_values, trends

def generate_prompts(batch_size, min_values, max_values, median_values, trends, mode="original"):
    prompts = []
    for i in range(batch_size):
        min_value = float(min_values[i].item())
        max_value = float(max_values[i].item())
        median_value = float(median_values[i].item())
        trend_value = float(trends[i].item())
        trend = 'upward' if trend_value > 0 else 'downward'
        if mode == "generic":
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                "The EEG signal shows certain temporal dynamics and amplitude variations."
            )
        else:
            prompt = (
                "Based on the provided EEG data, classify the underlying emotion. "
                f"Statistics: minimum value = {min_value:.2f}, maximum value = {max_value:.2f}, "
                f"median value = {median_value:.2f}. The overall trend of the data is {trend}."
            )
        prompts.append(prompt)
    return prompts

def cache_path_for_file(file_path, prompt_mode):
    cache_dir = Path(__file__).resolve().parents[1] / "prompt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{Path(file_path).stem}__{prompt_mode}__last_hidden_state_fp16.pt"

def main():
    ap = argparse.ArgumentParser()
    default_data_dir = Path(__file__).resolve().parents[1] / "data" / "SEED_chunks"
    if Path("/home/xiaoying/SEED-IV_chunks").exists():
        default_data_dir = Path("/home/xiaoying/SEED-IV_chunks")
    ap.add_argument("--data_dir", default=str(default_data_dir))
    ap.add_argument("--mode", default=os.getenv("PROMPT_MODE", "original"))
    default_bert = os.getenv("BERT_MODEL_DIR", "bert-base-uncased")
    local_bert = Path("/home/xiaoying/TPANet-main2/models/bert-base-uncased")
    if local_bert.exists():
        default_bert = str(local_bert)
    ap.add_argument("--bert", default=default_bert)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _validate_local_bert_weights(args.bert)
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    try:
        bert_model = BertModel.from_pretrained(args.bert, weights_only=False).to(device)
    except TypeError:
        bert_model = BertModel.from_pretrained(args.bert).to(device)
    bert_model.eval()
    for p in bert_model.parameters():
        p.requires_grad = False

    files = sorted(Path(args.data_dir).glob("*.npz"))
    print(f"Found {len(files)} files in {args.data_dir}")

    for fp in files:
        outp = cache_path_for_file(fp, args.mode)
        if outp.exists():
            print(f"[skip] {outp.name}")
            continue
        with np.load(fp, allow_pickle=True) as npz:
            all_data = npz["data"]
        X = torch.tensor(all_data, dtype=torch.float32)
        minv, maxv, medv, tr = generate_statistics(X)
        prompts = generate_prompts(X.size(0), minv, maxv, medv, tr, mode=args.mode)

        all_embeds = []
        with torch.no_grad():
            for s in range(0, X.size(0), args.batch_size):
                p_batch = prompts[s:s+args.batch_size]
                inp = tokenizer(
                    p_batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                emb = bert_model(**inp).last_hidden_state.detach().cpu().half()
                all_embeds.append(emb)
        cache = torch.cat(all_embeds, dim=0)
        torch.save(cache, outp)
        print(f"[ok] saved {outp} shape={tuple(cache.shape)} dtype={cache.dtype}")

if __name__ == "__main__":
    main()
