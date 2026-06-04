#!/usr/bin/env bash
set -euo pipefail

# 下载 ChatTTS 模型权重到 MODEL_DIR。
#
# ChatTTS 的自定义路径加载（source="custom"）要求目录下保留官方仓库的结构
# （asset/ 与 config/ 子目录），因此这里用 huggingface_hub 的 snapshot_download
# 整仓下载，保证目录结构与官方一致；大文件由 .gitignore/.dockerignore 控制不入库。
#
# 用法：
#   MODEL_DIR=./models bash scripts/fetch_models.sh
# 环境变量：
#   MODEL_DIR      目标目录（默认 repo/models）
#   HF_REPO_ID     HuggingFace 仓库（默认 2Noise/ChatTTS）
#   PYTHON         指定 Python 解释器（默认自动探测）

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models}"
HF_REPO_ID="${HF_REPO_ID:-2Noise/ChatTTS}"

# 选择 Python 解释器：优先 $PYTHON，其次项目虚拟环境 .venv，再退到 python3 / python。
# 这样在 pyenv（无裸 `python` shim）等本地环境也能正常工作；Docker slim 镜像内
# 则会落到 python3/python。
pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    echo "${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo "[fetch_models] 未找到 Python 解释器（python3 / python / .venv），请先安装或设置 PYTHON。" >&2
    exit 1
  fi
}

PYTHON_BIN="$(pick_python)"

mkdir -p "${MODEL_DIR}"

echo "[fetch_models] 使用解释器：${PYTHON_BIN}"
echo "[fetch_models] 从 ${HF_REPO_ID} 下载模型到 ${MODEL_DIR} ..."

# 确保 huggingface_hub 可用：缺失时给出明确提示，而不是抛 ImportError 栈。
if ! "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "[fetch_models] 当前解释器缺少 huggingface_hub，请先安装：" >&2
  echo "    ${PYTHON_BIN} -m pip install -r backend/requirements.txt" >&2
  echo "  或：${PYTHON_BIN} -m pip install 'huggingface_hub>=0.23'" >&2
  exit 1
fi

# 用 Python 调 huggingface_hub，避免手工拼大量大文件 URL；已存在的文件会被复用。
"${PYTHON_BIN}" - "$HF_REPO_ID" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, target = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo_id, local_dir=target)
print(f"[fetch_models] 模型已就绪：{path}")
PY

echo "[fetch_models] 完成。"
