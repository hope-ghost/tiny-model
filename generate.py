"""用训练好的模型续写小说：
  python generate.py --prompt "夜色渐深，" --tokens 300
参数：
  --prompt   开头文本（不给则从 <bos> 开始）
  --tokens   生成多少个 token
  --temp     温度，越高越有创意（0.7~1.0 常用）
  --top_k    只从概率最高的 k 个候选采样
"""
import os
import sys
import argparse
import torch
from tokenizers import Tokenizer

# Windows 控制台默认 GBK，强制 UTF-8 输出，保证中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import train_config as tc
from model import GPT

CKPT_PATH = os.path.join(tc.out_dir, "ckpt.pt")
TOKENIZER_PATH = os.path.join(tc.out_dir, "tokenizer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    args = ap.parse_args()

    device = tc.device if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 加载断点。checkpoint 里存了 model_config，保证结构一致
    ck = torch.load(CKPT_PATH, map_location=device)
    model = GPT(ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"已加载模型（训练到第 {ck['iter_num']} 步，val loss {ck['best_val']:.4f}）\n")

    # 把 prompt 编码成 token；空 prompt 用 <bos> 起头
    if args.prompt:
        start_ids = tokenizer.encode(args.prompt).ids
    else:
        bos = tokenizer.token_to_id("<bos>")
        start_ids = [bos if bos is not None else 0]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.tokens,
                           temperature=args.temp, top_k=args.top_k)
    text = tokenizer.decode(y[0].tolist())
    print("=" * 50)
    print(text)
    print("=" * 50)


if __name__ == "__main__":
    main()
