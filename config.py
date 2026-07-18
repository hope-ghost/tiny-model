"""集中管理所有超参数。改这里就够了，其它文件从这里读。

【重要】根据你的语料大小选预设，改下面的 PRESET 变量即可：
  数据量(txt总大小)   推荐预设      参数量    说明
  --------------------------------------------------------------
  < 5 MB             "micro"      ~7M       数据很少，用最小模型防过拟合
  5 ~ 30 MB          "tiny"       ~16M      小数据的稳妥选择
  30 ~ 150 MB        "small"      ~33M      数据较多才撑得起
  > 150 MB           "medium"     ~70M      大数据；默认设置峰值约 5GB 显存
  --------------------------------------------------------------
  选大了会过拟合（验证loss不降反升），选小了学不到位。拿不准就选小的。
  运行  python recommend.py  可根据你的数据自动推荐。
"""
from dataclasses import dataclass

# ============ 在这里选预设 ============
PRESET = "tiny"   # micro / tiny / small / medium
# ====================================

# 每档预设：(层数, Q头数, KV头数, 维度, dropout)。数据越少 → 模型越小、dropout越大
# n_kv_head < n_head 即启用 GQA（多个 Q 头共享 KV），省显存；n_head 须能被 n_kv_head 整除
PRESETS = {
    "micro":  dict(n_layer=4, n_head=4, n_kv_head=2, n_embd=256, dropout=0.2),
    "tiny":   dict(n_layer=6, n_head=6, n_kv_head=2, n_embd=384, dropout=0.2),
    "small":  dict(n_layer=8, n_head=8, n_kv_head=4, n_embd=512, dropout=0.1),
    "medium": dict(n_layer=12, n_head=10, n_kv_head=2, n_embd=640, dropout=0.1),
}


@dataclass
class ModelConfig:
    vocab_size: int = 16000   # 分词器词表大小，需与 prepare_data 一致
    block_size: int = 512     # 上下文长度（一次能看多少个 token）
    n_layer: int = PRESETS[PRESET]["n_layer"]
    n_head: int = PRESETS[PRESET]["n_head"]
    n_kv_head: int = PRESETS[PRESET]["n_kv_head"]   # GQA 的 KV 头数（< n_head 即启用）
    n_embd: int = PRESETS[PRESET]["n_embd"]
    dropout: float = PRESETS[PRESET]["dropout"]
    bias: bool = False        # 线性层是否用 bias，False 更快更稳（RMSNorm 无 bias）


@dataclass
class TrainConfig:
    # ---- 数据 ----
    data_dir: str = "data"        # 放小说 txt 的目录
    out_dir: str = "out"          # 保存模型和分词器的目录
    # ---- 优化器 ----
    batch_size: int = 16          # 单次前向的序列条数（显存不够就调小）
    grad_accum_steps: int = 8     # 梯度累积，有效 batch = 16*8 = 128
    learning_rate: float = 3e-4   # 峰值学习率
    min_lr: float = 3e-5          # 余弦退火到的最小学习率
    weight_decay: float = 0.1     # 权重衰减，越大正则越强（数据少可调到0.2）
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0        # 梯度裁剪，防梯度爆炸
    # ---- 训练时长 ----
    max_iters: int = 20000        # 总训练步数上限（通常早停会先触发）
    warmup_iters: int = 500       # 学习率预热步数
    # ---- 早停：验证loss连续不再改善就自动停，防止过拟合空练 ----
    early_stop: bool = True       # 是否启用早停
    patience: int = 5             # 验证loss连续几次没创新低就停（配合eval_interval）
    # ---- 评估与保存 ----
    eval_interval: int = 250      # 每多少步评估一次验证集
    eval_iters: int = 100         # 评估时采样多少个 batch 求平均 loss
    log_interval: int = 20        # 每多少步打印一次训练 loss
    # ---- 设备 ----
    device: str = "cuda"          # cuda / cpu
    dtype: str = "bfloat16"       # 4060Ti 支持；老卡用 float16
    compile_model: bool = False   # torch.compile 加速，Windows 上常不稳，默认关
    seed: int = 1337


model_config = ModelConfig()
train_config = TrainConfig()


# ============ 多项目支持 ============
import os
import json

PROJECTS_DIR = "projects"   # 所有项目的根目录


def project_paths(name):
    """返回某项目的各路径。项目名已在 dashboard 侧校验过合法性。"""
    root = os.path.join(PROJECTS_DIR, name)
    out = os.path.join(root, "out")
    return {
        "root": root,
        "data_dir": os.path.join(root, "data"),
        "out_dir": out,
        "meta_json": os.path.join(root, "project.json"),
        "tokenizer": os.path.join(out, "tokenizer.json"),
        "train_bin": os.path.join(out, "train.bin"),
        "val_bin": os.path.join(out, "val.bin"),
        "meta_pkl": os.path.join(out, "meta.pkl"),
        "ckpt": os.path.join(out, "ckpt.pt"),
        "log": os.path.join(out, "train_log.jsonl"),
        "pid": os.path.join(out, "train.pid"),
    }


def apply_preset(preset_name):
    """把指定预设应用到全局 model_config（脚本按项目覆盖时用）。"""
    p = PRESETS[preset_name]
    model_config.n_layer = p["n_layer"]
    model_config.n_head = p["n_head"]
    model_config.n_kv_head = p["n_kv_head"]
    model_config.n_embd = p["n_embd"]
    model_config.dropout = p["dropout"]


def use_project(name):
    """把 train_config 的 data_dir/out_dir 切到指定项目，并按 project.json 应用预设。
    脚本（prepare_data / train）启动时调用，实现"一份代码操作任意项目"。
    返回该项目的路径字典。供 dashboard 之外的命令行脚本使用。
    """
    paths = project_paths(name)
    train_config.data_dir = paths["data_dir"]
    train_config.out_dir = paths["out_dir"]
    # 读 project.json 里的预设；读不到就沿用全局 PRESET 默认值
    if os.path.exists(paths["meta_json"]):
        try:
            with open(paths["meta_json"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("preset") in PRESETS:
                apply_preset(meta["preset"])
        except (json.JSONDecodeError, OSError):
            pass
    return paths
