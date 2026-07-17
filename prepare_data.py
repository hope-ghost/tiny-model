"""数据预处理：读 data/ 下所有 txt → 训练 BPE 分词器 → 编码为二进制 token 文件。
运行一次即可：  python prepare_data.py
产物都放在 out/ 下：tokenizer.json、train.bin、val.bin、meta.pkl
"""
import os
import sys
import glob
import pickle
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Windows 控制台默认 GBK，强制 UTF-8 输出，保证中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from config import model_config as mc, train_config as tc, use_project


def read_corpus():
    """读取 data/ 下所有 .txt，合并成一个大字符串。"""
    files = glob.glob(os.path.join(tc.data_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(
            f"在 {tc.data_dir}/ 里没找到任何 .txt 文件，请先把小说文本放进去")
    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read())
        print(f"  读入 {f} ({os.path.getsize(f)/1e6:.1f} MB)")
    return "\n".join(texts)


def train_tokenizer(text):
    """用 BPE 训练一个中文分词器。BPE 能自动发现常用字/词组合。"""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # 按字节切分，中文每个字是多个字节，BPE 会把常见组合合并成一个 token
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=mc.vocab_size,
        special_tokens=["<unk>", "<bos>", "<eos>"],  # 未知/开头/结尾
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    # 分块喂给训练器，避免一次性占用过多内存（照顾 16G 内存）
    def chunks():
        step = 1_000_000  # 每块 100 万字符
        for i in range(0, len(text), step):
            yield text[i:i + step]
    print("训练 BPE 分词器中...")
    tokenizer.train_from_iterator(chunks(), trainer=trainer, length=len(text))
    tok_path = os.path.join(tc.out_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"分词器已保存到 {tok_path}，实际词表大小 {tokenizer.get_vocab_size()}")
    return tokenizer


def encode_and_save(text, tokenizer):
    """把全文编码成 token id，按 9:1 切成训练/验证集，存为 uint16 二进制。"""
    ids = tokenizer.encode(text).ids
    ids = np.array(ids, dtype=np.uint16)  # 词表<65536，用 uint16 省一半空间
    n = len(ids)
    split = int(n * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    train_ids.tofile(os.path.join(tc.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(tc.out_dir, "val.bin"))
    # 保存元信息，训练时读取。同时回填全局 mc，供 mark_prepared 记录
    mc.vocab_size = tokenizer.get_vocab_size()
    with open(os.path.join(tc.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": mc.vocab_size}, f)
    print(f"总 token 数 {n:,}  训练集 {len(train_ids):,}  验证集 {len(val_ids):,}")


def mark_prepared(project):
    """处理完成后更新 project.json：prepared=true、记录词表大小、清除重处理标记。"""
    if not project:
        return
    import json
    from config import project_paths
    mj = project_paths(project)["meta_json"]
    if not os.path.exists(mj):
        return
    try:
        with open(mj, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["prepared"] = True
        meta["needs_reprocess"] = False
        meta["vocab_size"] = mc.vocab_size
        with open(mj, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default=None,
                    help="项目名；不给则用 config.py 里的默认 data/out")
    args = ap.parse_args()
    if args.project:
        use_project(args.project)  # 切到该项目的 data_dir/out_dir，并应用其预设

    os.makedirs(tc.out_dir, exist_ok=True)
    print("== 1/3 读取语料 ==")
    text = read_corpus()
    print(f"语料总字符数 {len(text):,}")
    print("== 2/3 训练分词器 ==")
    tokenizer = train_tokenizer(text)
    print("== 3/3 编码并保存 ==")
    encode_and_save(text, tokenizer)
    mark_prepared(args.project)
    print("完成！接下来可以开始训练")


if __name__ == "__main__":
    main()
