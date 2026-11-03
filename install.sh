#!/usr/bin/env bash
# Installiert Music-Gen-UI in eine eigene Conda-Umgebung (Linux, NVIDIA).
#
#   ./install.sh                 Standard: Umgebung "music-gen-ui", CUDA 12.8, Modelle laden
#   ./install.sh --env NAME      anderer Umgebungsname
#   ./install.sh --fast          zusätzlich vLLM-Backend installieren
#   ./install.sh --no-models     Modelle nicht vorab herunterladen (passiert sonst beim 1. Song)
#   ./install.sh --mock          nur die Oberfläche, ohne PyTorch/YuE2 (zum Ausprobieren)
#   ./install.sh --cuda cu126    andere PyTorch-CUDA-Variante (Treiber zu alt für 12.8)
#   ./install.sh --yue2-ref REF  andere YuE2-Version (Git-Tag/Branch/Commit, z.B. main)
#
# Erneut ausführen = Update (Umgebung bleibt, Pakete werden aktualisiert).
set -euo pipefail

ENV_NAME="music-gen-ui"
CUDA="cu128"
TORCH_VERSION="2.10.0"
FAST=0
MODELS=1
MOCK=0
YUE2_REF="yue2-v0.1.6"  # official release; must match pyproject.toml
YUE2_REPO="https://github.com/multimodal-art-projection/YuE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --fast) FAST=1; shift ;;
    --no-models) MODELS=0; shift ;;
    --mock) MOCK=1; MODELS=0; shift ;;
    --yue2-ref) YUE2_REF="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unbekannte Option: $1 (siehe --help)" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$(readlink -f "$0")")"
step() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

# --- Conda finden -------------------------------------------------------------
find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then echo "$CONDA_EXE"; return; fi
  if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
  for dir in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda /opt/miniconda3; do
    if [[ -x "$dir/bin/conda" ]]; then echo "$dir/bin/conda"; return; fi
  done
  return 1
}
CONDA_BIN="$(find_conda)" || {
  echo "Conda wurde nicht gefunden. Miniconda installieren: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
}
# shellcheck disable=SC1091
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"

# --- Umgebung anlegen/aktualisieren -------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  step "Aktualisiere Conda-Umgebung '$ENV_NAME'"
  conda env update -n "$ENV_NAME" -f environment.yml
else
  step "Erstelle Conda-Umgebung '$ENV_NAME' (Python 3.12)"
  conda env create -n "$ENV_NAME" -f environment.yml
fi
set +u  # conda activation scripts may use unset variables
conda activate "$ENV_NAME"
set -u
# --- pip-Paketquellen prüfen ------------------------------------------------------
# Eine global eingetragene, nicht erreichbare Quelle (z.B. pypi.ngc.nvidia.com) bremst
# jede Installation mit 5 Wiederholungen pro Paket aus. Dann ignorieren wir für diesen
# Lauf die globale pip-Konfiguration (--isolated); PyPI und der PyTorch-Index reichen.
PIP=(python -m pip)
check_pip_sources() {
  local urls host bad=""
  urls="$( { python -m pip config list 2>/dev/null; printf '%s\n' "${PIP_INDEX_URL:-}" "${PIP_EXTRA_INDEX_URL:-}"; } \
           | { grep -Eo "https?://[^' \"]+" || true; } | sort -u)"  # no URLs configured is fine
  for url in $urls; do
    host="$(printf '%s' "$url" | sed -E 's#https?://([^/:@]+@)?([^/:]+).*#\2#')"
    python -c 'import socket, sys; socket.getaddrinfo(sys.argv[1], 443)' "$host" 2>/dev/null || bad="$bad $host"
  done
  if [[ -n "$bad" ]]; then
    echo "Hinweis: In deiner pip-Konfiguration ist eine nicht erreichbare Paketquelle eingetragen:$bad" >&2
    echo "         Für diese Installation wird die globale pip-Konfiguration ignoriert (pip --isolated)." >&2
    echo "         Dauerhaft entfernen: 'python -m pip config debug' zeigt die Datei," >&2
    echo "         dann z.B. 'python -m pip config unset global.extra-index-url' (oder PIP_EXTRA_INDEX_URL)." >&2
    PIP=(python -m pip --isolated)
    unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
  fi
}
check_pip_sources

"${PIP[@]}" install --upgrade pip

# --- Pakete -------------------------------------------------------------------
if [[ "$MOCK" == 1 ]]; then
  step "Installiere Oberfläche (ohne YuE2)"
  "${PIP[@]}" install -e ".[lua]"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    step "GPU"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "Hinweis: nvidia-smi nicht gefunden – ist der NVIDIA-Treiber installiert?" >&2
  fi
  step "Installiere PyTorch + torchaudio $TORCH_VERSION ($CUDA)"
  "${PIP[@]}" install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA"
  EXTRAS="yue2,lua,train"
  [[ "$FAST" == 1 ]] && EXTRAS="$EXTRAS,fast"
  step "Installiere Music-Gen-UI + YuE2 ($EXTRAS)"
  "${PIP[@]}" install -e ".[$EXTRAS]"
  # Always pin YuE2 explicitly: release and main share the version number 0.1.6,
  # so pip would otherwise keep whichever variant is already installed.
  step "Installiere YuE2-Version '$YUE2_REF'"
  "${PIP[@]}" install --force-reinstall --no-deps "yue2-infer @ git+$YUE2_REPO@$YUE2_REF"
  "${PIP[@]}" check || echo "Hinweis: Abhängigkeiten passen evtl. nicht exakt zu '$YUE2_REF' (siehe oben)." >&2
  step "YuE2-Umgebungscheck (yue2 doctor)"
  yue2 doctor || echo "Hinweis: 'yue2 doctor' meldet Probleme (siehe oben) – die UI startet trotzdem." >&2
fi

# --- Modelle ------------------------------------------------------------------
if [[ "$MODELS" == 1 ]]; then
  step "Lade YuE2-Modelle herunter (einmalig, mehrere GB; Ziel: Hugging-Face-Cache ${HF_HOME:-~/.cache/huggingface})"
  music-gen-ui-download
fi

step "Fertig"
echo "Starten mit:  ./start.sh            (dann http://127.0.0.1:7860 öffnen)"
echo "Im Netzwerk:  ./start.sh --host 0.0.0.0"
