"""训练主程序：  python train.py
支持 bf16 混合精度、梯度累积、余弦退火学习率、断点续训。
中断后再次运行会自动从 out/ckpt.pt 继续。
"""
import os
import sys
import math
import time
import pickle
import numpy as np
import torch

# Windows 控制台默认 GBK，强制 UTF-8 输出，保证中文不乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import argparse
from config import model_config as mc, train_config as tc, use_project
from model import GPT

# 这三个路径在 __main__ 里按 --project 重新派生（见文件末尾）
CKPT_PATH = os.path.join(tc.out_dir, "ckpt.pt")
LOG_PATH = os.path.join(tc.out_dir, "train_log.jsonl")   # 面板读取的指标日志
PID_PATH = os.path.join(tc.out_dir, "train.pid")         # 训练进程的 PID，面板据此判断是否在跑


def set_paths_for_out_dir():
    """根据当前 tc.out_dir 重新计算三个文件路径（切换项目后调用）。"""
    global CKPT_PATH, LOG_PATH, PID_PATH
    CKPT_PATH = os.path.join(tc.out_dir, "ckpt.pt")
    LOG_PATH = os.path.join(tc.out_dir, "train_log.jsonl")
    PID_PATH = os.path.join(tc.out_dir, "train.pid")


def log_metric(record):
    """把一条指标以 JSON 行追加写入日志，面板轮询这个文件画曲线。"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_batch(split):
    """随机取一个 batch。用 memmap 避免把整个数据集读进内存（照顾 16G 内存）。"""
    path = os.path.join(tc.out_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    # 随机选 batch_size 个起点
    ix = torch.randint(len(data) - mc.block_size, (tc.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + mc.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + mc.block_size].astype(np.int64)) for i in ix])
    # pin_memory + non_blocking 加速 CPU→GPU 传输
    if tc.device == "cuda":
        x, y = x.pin_memory().to(tc.device, non_blocking=True), y.pin_memory().to(tc.device, non_blocking=True)
    else:
        x, y = x.to(tc.device), y.to(tc.device)
    return x, y


def get_lr(it):
    """带预热的余弦退火学习率。"""
    if it < tc.warmup_iters:
        return tc.learning_rate * (it + 1) / (tc.warmup_iters + 1)
    if it > tc.max_iters:
        return tc.min_lr
    ratio = (it - tc.warmup_iters) / (tc.max_iters - tc.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)


@torch.no_grad()
def estimate_loss(model, ctx):
    """在训练/验证集上各采样若干 batch，求平均 loss。"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(tc.eval_iters)
        for k in range(tc.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def setup():
    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True   # 允许 TF32，A卡/N卡矩阵乘更快
    torch.backends.cudnn.allow_tf32 = True
    # 读取分词器实际词表大小，覆盖 config 里的默认值
    meta_path = os.path.join(tc.out_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            mc.vocab_size = pickle.load(f)["vocab_size"]
    device_type = "cuda" if "cuda" in tc.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[tc.dtype]
    ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
           if device_type == "cuda" else torch.no_grad())
    return device_type, ptdtype, ctx


def main():
    device_type, ptdtype, ctx = setup()
    model = GPT(mc).to(tc.device)
    print(f"模型参数量: {model.num_params()/1e6:.2f}M")
    optimizer = model.configure_optimizers(
        tc.weight_decay, tc.learning_rate, (tc.beta1, tc.beta2), device_type)
    # float16 需要 GradScaler 防下溢；bfloat16 不需要
    scaler = torch.amp.GradScaler(enabled=(tc.dtype == "float16"))

    iter_num, best_val = 0, 1e9
    # 断点续训：存在 ckpt 就加载
    if os.path.exists(CKPT_PATH):
        # weights_only=False：checkpoint 是我们自己存的（含 ModelConfig 对象），可信
        ck = torch.load(CKPT_PATH, map_location=tc.device, weights_only=False)
        try:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            iter_num, best_val = ck["iter_num"], ck["best_val"]
            print(f"发现断点 {CKPT_PATH}（iter {iter_num}），继续训练")
        except RuntimeError:
            # 结构不匹配：多半是改了 PRESET 预设。提示用户，不覆盖旧模型
            print("=" * 56)
            print("断点结构与当前模型不一致（你可能改了 config.py 的 PRESET）。")
            print(f"如要用新预设从头训练，请先删除旧断点：  del {CKPT_PATH}")
            print("（删除前可备份：旧模型是之前预设训练的结果）")
            print("=" * 56)
            sys.exit(1)
    elif os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)  # 全新训练：清掉旧日志，面板从头画

    log_metric({"type": "meta", "total_iters": tc.max_iters,
                "params_m": round(model.num_params() / 1e6, 2)})

    if tc.compile_model:
        model = torch.compile(model)

    no_improve = 0   # 验证loss连续几次没创新低，用于早停
    t0 = time.time()
    while iter_num <= tc.max_iters:
        lr = get_lr(iter_num)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 定期评估并保存最优模型
        if iter_num % tc.eval_interval == 0:
            losses = estimate_loss(model, ctx)
            print(f"[{iter_num}] train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")
            log_metric({"type": "eval", "iter": iter_num,
                        "train_loss": round(losses["train"], 4),
                        "val_loss": round(losses["val"], 4)})
            if losses["val"] < best_val:
                best_val = losses["val"]
                no_improve = 0   # 创了新低，计数清零
                raw = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({
                    "model": raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val": best_val,
                    "model_config": mc,
                }, CKPT_PATH)
                print(f"  ↳ 验证 loss 新低，已保存到 {CKPT_PATH}")
            else:
                no_improve += 1
                print(f"  ↳ 验证 loss 未改善（连续 {no_improve}/{tc.patience} 次），"
                      f"当前最优 {best_val:.4f}")
                # 早停：连续 patience 次没进步，说明开始过拟合，及时收手
                if tc.early_stop and no_improve >= tc.patience:
                    print(f"触发早停：验证 loss 已 {tc.patience} 次未创新低，"
                          f"停止训练。最优模型（val={best_val:.4f}）已保存。")
                    log_metric({"type": "earlystop", "iter": iter_num,
                                "best_val": round(best_val, 4)})
                    break

        # 梯度累积：累计多个 micro-batch 的梯度再更新一次
        for micro in range(tc.grad_accum_steps):
            X, Y = get_batch("train")
            with ctx:
                _, loss = model(X, Y)
                loss = loss / tc.grad_accum_steps  # 归一化
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % tc.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            lossf = loss.item() * tc.grad_accum_steps
            ms_per_it = dt * 1000 / max(tc.log_interval, 1)
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}, {ms_per_it:.0f}ms/it")
            log_metric({"type": "train", "iter": iter_num,
                        "loss": round(lossf, 4), "lr": lr,
                        "ms_per_it": round(ms_per_it, 1)})
        iter_num += 1

    log_metric({"type": "done", "iter": iter_num})
    print("训练结束。运行  python generate.py  看看效果")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default=None,
                    help="项目名；不给则用 config.py 里的默认 data/out")
    args = ap.parse_args()
    if args.project:
        use_project(args.project)     # 切 data_dir/out_dir + 应用该项目预设
        set_paths_for_out_dir()       # 三个文件路径跟着 out_dir 重新派生

    # 写入自己的 PID，面板据此判断训练是否在跑（跨面板重启依然有效）
    os.makedirs(tc.out_dir, exist_ok=True)
    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))
    try:
        main()
    except KeyboardInterrupt:
        print("训练被中断。")
    finally:
        # 正常结束或 Ctrl+C 时清理 PID 文件（被强杀时清理不了，面板会用存活检测兜底）
        if os.path.exists(PID_PATH):
            os.remove(PID_PATH)
