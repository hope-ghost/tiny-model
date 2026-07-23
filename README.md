# tiny-model：本地训练中文小说续写模型

在本地 NVIDIA 显卡上，从零训练一个 **7M～70M 参数** 的中文小说续写 GPT（语言模型 / LLM），学会续写你投喂的那类小说。

适合想自己做「网文 / 玄幻 / 言情风格续写」、又没有大显存的人：**训练中文小说续写模型**、文本生成、本地小模型，一条龙。

默认配置（bf16、`batch_size=16`）下，训练峰值显存大约 **2～5 GB**（见下表），**6GB 及以上**即可跑满全部预设；不需要云端 GPU，不需要 HuggingFace 账号，**上传 txt → 点几下 → 看曲线 → 试续写**，全程可在浏览器里完成。

---

## 亮点

| 特性 | 说明 |
|------|------|
| **可视化面板** | 上传语料、选模型、训练、看 loss 曲线、在线续写，一条龙 |
| **多项目管理** | 玄幻、言情、试验各开一个项目，数据与模型互不干扰 |
| **四档智能预设** | 按语料大小自动推荐 `micro` / `tiny` / `small` / `medium`，防过拟合 |
| **早停 + 断点续训** | 验证 loss 不再改善自动停；随时中断，下次接着训 |
| **中文 BPE 分词** | 自动学字词组合，无需自己造词表 |
| **现代小模型架构** | RoPE 位置编码 + RMSNorm + SwiGLU + GQA，收敛更稳、更省显存 |
| **省显存设计** | bf16、梯度累积、memmap 读数据；默认设置约 2～5 GB 即可训 |
| **代码精简可读** | 核心就 `model.py` + `train.py`，方便学习和改 |

---

## 快速开始（推荐：面板）

### 1. 安装环境

**Python 3.10+**，NVIDIA 显卡 + CUDA。

```powershell
# 先装 GPU 版 PyTorch（按你的 CUDA 版本选，查 nvidia-smi）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 再装其余依赖
pip install -r requirements.txt
pip install flask   # 面板需要
```

或直接双击 `1_安装依赖.bat`（已内置 PyTorch + Flask 安装）。

验证 GPU：

```powershell
python smoke_test.py
```

### 2. 启动面板

双击 `2_启动面板.bat`，或：

```powershell
python dashboard.py
```

浏览器打开 **http://127.0.0.1:5000**

### 3. 三步走

1. **新建项目** → 上传同类小说 `.txt`（UTF-8，可多文件、可多次追加）
2. **选模型大小**（系统会按数据量自动推荐）→ 点 **开始处理**
3. **开始训练** → 看实时曲线 → 训练一段时间后点 **生成** 试续写

---

## 模型预设

语料越少，模型应越小；选大了容易过拟合，选小了学不到位。

| 预设 | 参数量 | 推荐语料量 | 层 / 头 / 维度 | 训练峰值显存* |
|------|--------|------------|----------------|---------------|
| `micro` | ~7M | < 5 MB | 4 / 4 / 256 | ~2 GB |
| `tiny` | ~17M | 5～30 MB | 6 / 6 / 384 | ~2.5 GB |
| `small` | ~33M | 30～150 MB | 8 / 8 / 512 | ~3 GB |
| `medium` | ~70M | > 150 MB | 12 / 10 / 640 | ~5 GB |

\*在 RTX 4060 Ti 上实测：`bfloat16`、`batch_size=16`、`block_size=512`、含 AdamW；`nvidia-smi` 因显存池缓存可能略高一点。把 `batch_size` 调到 8/4 可再明显降占用。

拿不准就选小的。也可运行 `python recommend.py` 查看推荐（命令行模式下扫描 `data/` 目录）。

---

## 命令行用法（可选）

面板底层调用的就是下面这些脚本，也支持直接命令行操作。

```powershell
# 数据处理（训练 BPE 分词器 + 编码）
python prepare_data.py --project 我的项目名

# 训练（支持断点续训、早停）
python train.py --project 我的项目名

# 续写（默认读写根目录 data/ 与 out/，多项目建议用面板生成）
python generate.py --prompt "夜色渐深，" --tokens 300 --temp 0.8
```

不传 `--project` 时，使用 `config.py` 里默认的 `data/`、`out/` 路径（单项目 / 旧版用法）。

---

## 项目结构

```
tiny-model/
├── dashboard.py          # 面板后端
├── dashboard.html        # 面板前端
├── config.py             # 全局超参与预设定义
├── model.py              # GPT 模型
├── prepare_data.py       # 数据预处理
├── train.py              # 训练
├── generate.py           # 命令行续写
├── recommend.py          # 按语料推荐预设
├── smoke_test.py         # 环境自检
├── projects/             # 多项目目录（面板创建）
│   └── <项目名>/
│       ├── project.json  # 预设、处理状态等元信息
│       ├── data/         # 上传的 txt
│       └── out/          # 分词器、训练数据、模型、日志
├── 1_安装依赖.bat
└── 2_启动面板.bat
```

从旧版（根目录 `data/` + `out/`）迁移到多项目，可运行：

```powershell
python migrate_to_projects.py
```

---

## 效果预期

**优先看验证 loss（val loss）**，不要只盯训练 loss。

| 验证 loss | 大致水平 |
|-----------|----------|
| ~9.7 | 未训练（随机基线） |
| 5～6 | 及格，句子开始通顺 |
| 4.5 左右 | 不错，可实用续写 |
| 4.0 左右 | 对本项目而言已经很好 |

若 **train loss 很低但 val loss 反弹**（如 train 1.x、val 6+），说明过拟合——换更小预设，或加语料，相信早停保存的最优 checkpoint。

生成质量最终以 **实际续写** 为准。小模型只会续写，不会对答，这是正常的。

### 提升效果的建议

1. **语料**：同风格、够量、干净，比盲目加大模型更有效
2. **预设**：与数据量匹配，有过拟合就降一档
3. **生成**：`temp` 0.7～0.85；给 20～50 字 prompt 作风格样例

---

## 调参速查（`config.py`）

| 想要 | 改什么 |
|------|--------|
| 显存不够（OOM，多见于 4GB 卡） | 调小 `batch_size`（如 8、4） |
| 训练更久 | 加大 `max_iters` |
| 更早/更晚停 | 调整 `patience`（配合 `eval_interval`） |
| 生成更保守 | 降低 `--temp`（如 0.6） |
| 生成更有创意 | 提高 `--temp`（如 1.0），或调 `top_k` |

**换预设后需重新处理数据并从头训练**（旧 `ckpt.pt` 结构不兼容，需删除后重来）。

---

## 技术栈

- PyTorch + bf16 混合精度
- 现代小模型架构：**RoPE**（旋转位置编码）、**RMSNorm**、**SwiGLU** 前馈、**GQA**（分组查询注意力）
- AdamW、余弦退火学习率、梯度累积与裁剪
- Flash Attention（`scaled_dot_product_attention`）
- HuggingFace `tokenizers`（BPE）
- Flask 面板

> 架构说明：早期版本为 GPT-2 经典结构（绝对位置嵌入 + LayerNorm + GELU + MHA），现已升级为上述现代组件。若你有旧版本训练的 `ckpt.pt`，结构不兼容，需删除后重新训练（`train.py` 会自动检测并提示）。

---

## 许可

个人学习与研究使用。语料请确保你有权使用。
