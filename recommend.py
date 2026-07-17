"""根据 data/ 里的语料大小，推荐该用哪档模型预设。
用法：  python recommend.py
"""
import os
import sys
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import train_config as tc, PRESET

# (上限MB, 预设名, 参数量, 说明)
TABLE = [
    (5,    "micro",  "~7M",  "数据很少，用最小模型防过拟合"),
    (30,   "tiny",   "~16M", "小数据的稳妥选择"),
    (150,  "small",  "~33M", "数据较多才撑得起"),
    (1e9,  "medium", "~70M", "大数据 + 8G显存上限附近"),
]


def main():
    files = glob.glob(os.path.join(tc.data_dir, "*.txt"))
    if not files:
        print(f"data/ 里没有 txt 文件。请先放入小说文本。")
        return
    total = sum(os.path.getsize(f) for f in files)
    mb = total / 1e6
    print(f"检测到 {len(files)} 个文件，共 {mb:.1f} MB")
    for limit, name, params, desc in TABLE:
        if mb < limit:
            print(f"\n推荐预设：\"{name}\"（{params} 参数）—— {desc}")
            print(f"请打开 config.py，把 PRESET 改成  \"{name}\"")
            if name != PRESET:
                print(f"（当前是 \"{PRESET}\"，建议修改）")
            else:
                print(f"（当前已是 \"{PRESET}\"，无需改动）")
            break
    print("\n提示：换预设后需重新训练（旧断点结构不同，删掉 out/ckpt.pt 重来）")


if __name__ == "__main__":
    main()
