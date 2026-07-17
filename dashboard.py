"""可视化控制面板后端（Flask）——多项目版。
启动后浏览器打开 http://127.0.0.1:5000
功能：项目管理、数据上传、一键处理、按项目训练/续写、实时 loss 曲线。
"""
import os
import re
import sys
import json
import glob
import pickle
import subprocess
import threading
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import project_paths, PRESETS, PROJECTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 单次上传上限 500MB

_proc_lock = threading.Lock()
_gen_cache = {}   # {project_name: (model, tokenizer, ckpt_mtime)}

# 数据量 → 推荐预设 (上限MB, 预设名)
_RECO_TABLE = [(5, "micro"), (30, "tiny"), (150, "small"), (1e9, "medium")]
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")   # 合法项目名


def valid_name(name):
    return bool(name and NAME_RE.match(name))


def _pid_alive(pid):
    """检查 PID 是否为存活的 python 进程（Windows tasklist）。"""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5).stdout
        return str(pid) in out and "python" in out.lower()
    except Exception:
        return False


def _read_pid(pid_path):
    """读 PID 文件；进程已死则清理陈旧文件，返回存活 PID 或 None。"""
    if not os.path.exists(pid_path):
        return None
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return None
    if _pid_alive(pid):
        return pid
    try:
        os.remove(pid_path)
    except OSError:
        pass
    return None


def load_meta(name):
    """读项目 project.json，失败返回 None。"""
    mj = project_paths(name)["meta_json"]
    if not os.path.exists(mj):
        return None
    try:
        with open(mj, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_meta(name, meta):
    mj = project_paths(name)["meta_json"]
    with open(mj, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def recommend_preset(mb):
    for limit, preset in _RECO_TABLE:
        if mb < limit:
            return preset
    return "medium"


@app.route("/")
def index():
    return send_from_directory(HERE, "dashboard.html")


def project_status(name):
    """汇总一个项目的状态：数据量、是否已处理、有无模型、是否在训练。"""
    p = project_paths(name)
    meta = load_meta(name) or {}
    data_files = []
    total = 0
    if os.path.isdir(p["data_dir"]):
        for f in glob.glob(os.path.join(p["data_dir"], "*.txt")):
            sz = os.path.getsize(f)
            data_files.append({"name": os.path.basename(f), "size": sz})
            total += sz
    return {
        "name": name,
        "preset": meta.get("preset", "tiny"),
        "data_files": data_files,
        "data_mb": round(total / 1e6, 1),
        "prepared": os.path.exists(p["train_bin"]),
        "needs_reprocess": meta.get("needs_reprocess", False),
        "has_ckpt": os.path.exists(p["ckpt"]),
        "training": _read_pid(p["pid"]) is not None,
        "preparing": _read_pid(p["pid"] + ".prep") is not None,
    }


@app.route("/api/projects")
def api_projects():
    """列出所有项目及其状态。"""
    names = []
    if os.path.isdir(PROJECTS_DIR):
        names = [d for d in os.listdir(PROJECTS_DIR)
                 if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    return jsonify({"projects": [project_status(n) for n in sorted(names)]})


@app.route("/api/projects", methods=["POST"])
def api_create_project():
    """新建项目：只需项目名。预设先留空，等上传数据后按数据量自动推荐。"""
    data = request.get_json(force=True) or {}
    name = (data.get("name") or "").strip()
    if not valid_name(name):
        return jsonify({"ok": False, "msg": "项目名只能含字母数字下划线中划线，长度1-40"}), 400
    p = project_paths(name)
    if os.path.exists(p["root"]):
        return jsonify({"ok": False, "msg": "项目已存在"}), 400
    os.makedirs(p["data_dir"], exist_ok=True)
    os.makedirs(p["out_dir"], exist_ok=True)
    # preset_auto=True 表示预设仍由数据量自动决定，直到用户手动改过
    save_meta(name, {"name": name, "preset": "tiny", "preset_auto": True,
                     "created": "new", "data_files": [], "prepared": False,
                     "needs_reprocess": False, "vocab_size": 16000})
    return jsonify({"ok": True, "name": name})


@app.route("/api/projects/<name>/upload", methods=["POST"])
def api_upload(name):
    """上传 txt 到项目 data/。多文件、追加。已处理过的项目标记需重处理。"""
    if not valid_name(name) or not os.path.isdir(project_paths(name)["root"]):
        return jsonify({"ok": False, "msg": "项目不存在"}), 404
    p = project_paths(name)
    files = request.files.getlist("files")
    if not files:
        return jsonify({"ok": False, "msg": "没有收到文件"}), 400
    saved = []
    for fs in files:
        fn = secure_filename(fs.filename or "")
        if not fn.lower().endswith(".txt"):
            continue  # 只收 txt
        fs.save(os.path.join(p["data_dir"], fn))
        saved.append(fn)
    if not saved:
        return jsonify({"ok": False, "msg": "只接受 .txt 文件"}), 400
    meta = load_meta(name) or {}
    # 若此前已处理过，追加数据后需重新处理
    if os.path.exists(p["train_bin"]):
        meta["needs_reprocess"] = True
    # 用户没手动改过预设时，按当前数据总量自动推荐并选中
    if meta.get("preset_auto", True):
        total = sum(os.path.getsize(f)
                    for f in glob.glob(os.path.join(p["data_dir"], "*.txt")))
        meta["preset"] = recommend_preset(total / 1e6)
    save_meta(name, meta)
    return jsonify({"ok": True, "saved": saved,
                    "preset": meta.get("preset"),
                    "needs_reprocess": meta.get("needs_reprocess", False)})


@app.route("/api/projects/<name>/recommend")
def api_recommend(name):
    """按当前数据总量推荐预设。"""
    if not valid_name(name):
        return jsonify({"ok": False, "msg": "非法项目名"}), 400
    p = project_paths(name)
    total = 0
    if os.path.isdir(p["data_dir"]):
        for f in glob.glob(os.path.join(p["data_dir"], "*.txt")):
            total += os.path.getsize(f)
    mb = total / 1e6
    return jsonify({"ok": True, "data_mb": round(mb, 1),
                    "preset": recommend_preset(mb)})


@app.route("/api/projects/<name>/preset", methods=["POST"])
def api_set_preset(name):
    """更新项目预设（处理/训练前可改）。"""
    data = request.get_json(force=True) or {}
    preset = data.get("preset")
    if preset not in PRESETS:
        return jsonify({"ok": False, "msg": "无效预设"}), 400
    meta = load_meta(name)
    if meta is None:
        return jsonify({"ok": False, "msg": "项目不存在"}), 404
    meta["preset"] = preset
    meta["preset_auto"] = False   # 用户手动选过，之后上传不再自动改
    save_meta(name, meta)
    return jsonify({"ok": True})


@app.route("/api/projects/<name>/prepare", methods=["POST"])
def api_prepare(name):
    """后台跑 prepare_data.py --project name。用 .prep PID 文件跟踪。"""
    if not valid_name(name) or not os.path.isdir(project_paths(name)["root"]):
        return jsonify({"ok": False, "msg": "项目不存在"}), 404
    p = project_paths(name)
    prep_pid = p["pid"] + ".prep"
    with _proc_lock:
        if _read_pid(prep_pid) is not None:
            return jsonify({"ok": False, "msg": "正在处理中"}), 400
        # 数据检查
        if not glob.glob(os.path.join(p["data_dir"], "*.txt")):
            return jsonify({"ok": False, "msg": "请先上传 txt 数据"}), 400
        log = open(os.path.join(p["out_dir"], "prepare.log"), "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(HERE, "prepare_data.py"), "--project", name],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        with open(prep_pid, "w") as f:
            f.write(str(proc.pid))
    return jsonify({"ok": True, "pid": proc.pid})


@app.route("/api/projects/<name>/prepare_log")
def api_prepare_log(name):
    """返回处理日志的尾部内容，供前端实时显示进度。"""
    if not valid_name(name):
        return jsonify({"ok": False, "msg": "非法项目名"}), 400
    log_path = os.path.join(project_paths(name)["out_dir"], "prepare.log")
    text = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()[-4000:]   # 只取尾部，避免过大
        except OSError:
            pass
    return jsonify({"ok": True, "log": text})


def _require_project():
    """从 query/body 取项目名并校验存在。返回 (name, paths) 或 (None, error_response)。"""
    name = request.args.get("project") or (request.get_json(silent=True) or {}).get("project")
    if not valid_name(name) or not os.path.isdir(project_paths(name)["root"]):
        return None, (jsonify({"ok": False, "msg": "项目不存在或未指定"}), 404)
    return name, project_paths(name)


@app.route("/api/metrics")
def api_metrics():
    """读取指定项目的训练日志，返回指标点。"""
    name, p = _require_project()
    if name is None:
        return p
    train_pts, eval_pts, meta, done = [], [], {}, False
    if os.path.exists(p["log"]):
        with open(p["log"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type")
                if t == "train":
                    train_pts.append(r)
                elif t == "eval":
                    eval_pts.append(r)
                elif t == "meta":
                    meta = r
                elif t in ("done", "earlystop"):
                    done = True
    return jsonify({"train": train_pts, "eval": eval_pts, "meta": meta, "done": done})


@app.route("/api/status")
def api_status():
    """返回指定项目的训练状态。"""
    name, p = _require_project()
    if name is None:
        return p
    pid = _read_pid(p["pid"])
    prep = _read_pid(p["pid"] + ".prep")
    return jsonify({"training": pid is not None, "pid": pid,
                    "preparing": prep is not None,
                    "has_ckpt": os.path.exists(p["ckpt"]),
                    "prepared": os.path.exists(p["train_bin"])})


@app.route("/api/train/start", methods=["POST"])
def api_train_start():
    name, p = _require_project()
    if name is None:
        return p
    with _proc_lock:
        if _read_pid(p["pid"]) is not None:
            return jsonify({"ok": False, "msg": "训练已在运行中"}), 400
        if not os.path.exists(p["train_bin"]):
            return jsonify({"ok": False, "msg": "请先处理数据"}), 400
        proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "train.py"), "--project", name],
            cwd=HERE)
    return jsonify({"ok": True, "pid": proc.pid})


@app.route("/api/train/stop", methods=["POST"])
def api_train_stop():
    name, p = _require_project()
    if name is None:
        return p
    with _proc_lock:
        pid = _read_pid(p["pid"])
        if pid is None:
            return jsonify({"ok": False, "msg": "训练未在运行"}), 400
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                           capture_output=True, timeout=10)
        except Exception as e:
            return jsonify({"ok": False, "msg": f"停止失败: {e}"}), 500
        if os.path.exists(p["pid"]):
            try:
                os.remove(p["pid"])
            except OSError:
                pass
    return jsonify({"ok": True})


def _load_gen_model(name):
    """按需加载某项目的生成模型；按 ckpt mtime 缓存，训练中用 CPU。"""
    import torch, time
    from tokenizers import Tokenizer
    from model import GPT
    p = project_paths(name)
    mtime = os.path.getmtime(p["ckpt"])
    cached = _gen_cache.get(name)
    if cached and cached[2] == mtime:
        return cached[0], cached[1]

    training = _read_pid(p["pid"]) is not None
    device = "cpu" if training else ("cuda" if torch.cuda.is_available() else "cpu")

    last_err = None
    for _ in range(3):
        try:
            ck = torch.load(p["ckpt"], map_location=device, weights_only=False)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.5)
    else:
        raise RuntimeError(f"模型读取失败（训练正在写入，稍后再试）: {last_err}")

    tok = Tokenizer.from_file(p["tokenizer"])
    m = GPT(ck["model_config"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    _gen_cache[name] = (m, tok, mtime)
    return m, tok


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """用指定项目的模型续写。异常一律转 JSON。"""
    try:
        name, p = _require_project()
        if name is None:
            return p
        if not os.path.exists(p["ckpt"]):
            return jsonify({"ok": False, "msg": "该项目还没有训练好的模型"}), 400
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "")
        tokens = int(data.get("tokens", 200))
        temp = float(data.get("temp", 0.8))
        top_k = int(data.get("top_k", 200))
        import torch
        model, tok = _load_gen_model(name)
        device = next(model.parameters()).device
        if prompt:
            ids = tok.encode(prompt).ids
        else:
            bos = tok.token_to_id("<bos>")
            ids = [bos if bos is not None else 0]
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=tokens, temperature=temp, top_k=top_k)
        return jsonify({"ok": True, "text": tok.decode(y[0].tolist())})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "msg": str(e), "detail": traceback.format_exc()}), 500


if __name__ == "__main__":
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    print("面板已启动，请在浏览器打开:  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
