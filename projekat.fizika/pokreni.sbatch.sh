#!/bin/bash
#SBATCH --job-name=ekstrakcija
#SBATCH --partition=nordeus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.err

set -euo pipefail

# Učitavanje Miniconda modula
source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniconda

# Putanje za video zapise i izlazne podatke
export VIDEO_DIR="${SLURM_SUBMIT_DIR}/videos"
export OUTPUT_DIR="${SLURM_SUBMIT_DIR}/obradjene_koordinate"
mkdir -p "$OUTPUT_DIR"

echo "=== Pokrećem obradu koordinata na klasteru ==="

# Pokretanje Python skripte kroz okruženje fizika-env
conda run --no-capture-output -n fizika-env python "${SLURM_SUBMIT_DIR}/vadjenjekoordinata.py"

echo "=== Posao je uspešno završen ==="