"""一次性迁移脚本：把旧的 data/ + out/ 转成一个项目。
用法：  python migrate_to_projects.py [--name 项目名]
不给 --name 时用中性名 my_novel（只是个普通项目，无特殊地位）。
安全：只拷贝不删除，原 data/ out/ 保留。迁移后可手动清理。
"""
import os
import sys
import json
import shutil
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import project_paths, PRESET


def detect_preset():
    """尽量探测旧模型用的预设：先看备份 ckpt 的结构，否则用 config 里的 PRESET。"""
    backup = os.path.join("out", "ckpt_small_backup.pt")
    ckpt = os.path.join("out", "ckpt.pt")
    for path in (ckpt, backup):
        if os.path.exists(path):
            try:
                import torch
                from config import PRESETS
                ck = torch.load(path, map_location="cpu", weights_only=False)
                mcfg = ck.get("model_config")
                if mcfg is not None:
                    # 按 n_embd 反查预设名
                    for name, p in PRESETS.items():
                        if p["n_embd"] == mcfg.n_embd and p["n_layer"] == mcfg.n_layer:
                            return name
            except Exception:
                pass
    return PRESET   # 探测不到就用当前默认


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="my_novel", help="迁移成的项目名（普通项目，可任意取）")
    args = ap.parse_args()
    name = args.name

    if not os.path.isdir("data") and not os.path.isdir("out"):
        print("没有找到旧的 data/ 或 out/ 目录，无需迁移。")
        return

    paths = project_paths(name)
    if os.path.exists(paths["root"]):
        print(f"projects/{name}/ 已存在，跳过迁移（避免覆盖）。")
        return

    os.makedirs(paths["data_dir"], exist_ok=True)
    os.makedirs(paths["out_dir"], exist_ok=True)

    # 拷贝数据文件
    data_files = []
    if os.path.isdir("data"):
        for fn in os.listdir("data"):
            if fn.endswith(".txt"):
                src = os.path.join("data", fn)
                shutil.copy2(src, os.path.join(paths["data_dir"], fn))
                data_files.append({"name": fn, "size": os.path.getsize(src)})
                print(f"  拷贝数据 {fn}")

    # 拷贝 out/ 产物（tokenizer/bin/meta/ckpt 等），跳过日志/临时文件
    prepared = False
    if os.path.isdir("out"):
        for fn in os.listdir("out"):
            if fn.endswith((".log",)):
                continue
            src = os.path.join("out", fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(paths["out_dir"], fn))
                print(f"  拷贝产物 {fn}")
        prepared = os.path.exists(paths["train_bin"])

    # 读词表大小（若有 meta.pkl）
    vocab_size = 16000
    if os.path.exists(paths["meta_pkl"]):
        try:
            import pickle
            with open(paths["meta_pkl"], "rb") as f:
                vocab_size = pickle.load(f)["vocab_size"]
        except Exception:
            pass

    preset = detect_preset()
    meta = {
        "name": name,
        "preset": preset,
        "created": "migrated",
        "data_files": data_files,
        "prepared": prepared,
        "needs_reprocess": False,
        "vocab_size": vocab_size,
    }
    with open(paths["meta_json"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n迁移完成 → projects/{name}/")
    print(f"  预设 {preset}，已处理={prepared}，数据文件 {len(data_files)} 个")
    print("原 data/ 和 out/ 已保留未删。确认无误后可手动删除它们。")


if __name__ == "__main__":
    main()
