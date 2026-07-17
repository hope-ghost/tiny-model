"""环境自检：验证 GPU、模型构建、前向、反向、生成是否都能跑通。
不需要真实数据，用随机 token 测。跑完会自动清理。"""
import torch
from config import model_config as mc, train_config as tc
from model import GPT

print("torch", torch.__version__, "| CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
mc.vocab_size = 16000
model = GPT(mc).to(device)
print(f"参数量: {model.num_params()/1e6:.2f}M")

# 造一个随机 batch 测前向+反向
x = torch.randint(0, mc.vocab_size, (2, mc.block_size), device=device)
y = torch.randint(0, mc.vocab_size, (2, mc.block_size), device=device)
with torch.amp.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
    logits, loss = model(x, y)
print(f"前向 OK, 初始 loss {loss.item():.4f} (理论≈{torch.log(torch.tensor(mc.vocab_size*1.0)):.2f})")
loss.backward()
print("反向 OK")

# 测生成
model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(start, max_new_tokens=10)
print(f"生成 OK, 输出 {out.shape[1]} 个 token")

if device == "cuda":
    print(f"显存占用: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("\n[OK] 环境自检全部通过，可以开始准备数据了")
