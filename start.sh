#!/usr/bin/env bash
# Startet Music-Gen-UI in der Conda-Umgebung. Alle Argumente gehen an music-gen-ui,
# z.B.  ./start.sh --host 0.0.0.0 --port 7860 --preload --fp8
# Andere Umgebung:  MGU_CONDA_ENV=meine-env ./start.sh
set -euo pipefail
ENV_NAME="${MGU_CONDA_ENV:-music-gen-ui}"
cd "$(dirname "$(readlink -f "$0")")"

CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  for dir in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda; do
    [[ -x "$dir/bin/conda" ]] && CONDA_BIN="$dir/bin/conda" && break
  done
fi
[[ -n "$CONDA_BIN" ]] || { echo "Conda nicht gefunden – zuerst ./install.sh ausführen." >&2; exit 1; }
# shellcheck disable=SC1091
source "$("$CONDA_BIN" info --base)/etc/profile.d/conda.sh"
set +u  # conda activation scripts may use unset variables
conda activate "$ENV_NAME" || { echo "Umgebung '$ENV_NAME' fehlt – ./install.sh ausführen." >&2; exit 1; }
exec music-gen-ui "$@"
