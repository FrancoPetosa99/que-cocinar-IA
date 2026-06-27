#!/usr/bin/env bash
# Bootstrap and run Qué Cocinar IA (venv, deps, ingest, HF model, Gradio UI).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

step() { printf '\n▶ %s\n' "$1"; }

# --- virtualenv ---
if [[ ! -d "$VENV" ]]; then
  step "Creating virtual environment"
  python3 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

step "Installing dependencies"
"$PIP" install -q -r requirements.txt

# --- .env ---
if [[ ! -f "$ROOT/.env" ]]; then
  step "Creating .env from .env.example"
  cp "$ROOT/.env.example" "$ROOT/.env"
fi

# Load env for model / provider checks
set -a
# shellcheck disable=SC1091
source "$ROOT/.env"
set +a

LLM_PROVIDER="${LLM_PROVIDER:-huggingface}"
HF_BACKEND="${HF_BACKEND:-local}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HF_MODEL_CACHE_DIR="${HF_MODEL_CACHE_DIR:-models}"
SQLITE_PATH="${SQLITE_PATH:-data/recipes.db}"
CHROMA_DIR="${CHROMA_DIR:-chroma_db}"

# Resolve relative paths against project root
[[ "$SQLITE_PATH" != /* ]] && SQLITE_PATH="$ROOT/$SQLITE_PATH"
[[ "$CHROMA_DIR" != /* ]] && CHROMA_DIR="$ROOT/$CHROMA_DIR"
[[ "$HF_MODEL_CACHE_DIR" != /* ]] && HF_MODEL_CACHE_DIR="$ROOT/$HF_MODEL_CACHE_DIR"

# --- recipe data ---
CSV="$ROOT/data/recipes.csv"
if [[ ! -f "$CSV" ]]; then
  echo "Error: missing $CSV — add the recipes CSV before starting." >&2
  exit 1
fi

needs_ingest=false
if [[ ! -f "$SQLITE_PATH" ]]; then
  needs_ingest=true
elif [[ ! -d "$CHROMA_DIR" ]] || [[ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null || true)" ]]; then
  needs_ingest=true
fi

if $needs_ingest; then
  step "Indexing recipes (SQLite + Chroma)"
  "$PYTHON" data_preprocessing/ingest.py
else
  step "Recipe databases already present — skipping ingest"
fi

# --- local Hugging Face model ---
if [[ "$LLM_PROVIDER" == "huggingface" && "$HF_BACKEND" == "local" ]]; then
  MODEL_FOLDER="${LLM_MODEL##*/}"
  MODEL_DIR="$HF_MODEL_CACHE_DIR/$MODEL_FOLDER"

  if [[ ! -d "$MODEL_DIR" ]] || [[ -z "$(ls -A "$MODEL_DIR" 2>/dev/null || true)" ]]; then
    step "Downloading local LLM: $LLM_MODEL (this may take a few minutes)"
    "$PYTHON" scripts/download_hf_model.py "$LLM_MODEL"
  else
    step "Local LLM already downloaded — skipping"
  fi
fi

# --- launch ---
step "Starting Gradio app at http://127.0.0.1:7860"
exec "$PYTHON" frontend/app.py
