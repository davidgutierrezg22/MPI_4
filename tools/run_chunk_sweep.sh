#!/bin/sh
# Supplementary experiment: effect of the chunk size of mpi2.py (dynamic strategy).
#   docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 sh /app/tools/run_chunk_sweep.sh
# For each scenario (bind = /app, local = copy inside the container) runs
# 1 warm-up + RUNS interleaved rounds of: mpi1, mpi2 --chunk C for C in CHUNKS.
set -eu
APP_DIR="/app"
NP="${NP:-9}"
RUNS="${RUNS:-5}"
CHUNKS="${CHUNKS:-1 4 16 64}"
OUT="$APP_DIR/evidence/chunk_sweep"
MPI="mpirun --allow-run-as-root --oversubscribe -np $NP python"
rm -rf "$OUT"; mkdir -p "$OUT"

round() {  # scenario dir tag
  sc="$1"; dir="$2"; tag="$3"
  $MPI "$dir/mpi1.py" > "$OUT/$sc/mpi1_$tag.log" 2>&1
  printf "  mpi1        %s\n" "$(grep -E 'Tiempo|LI=' "$OUT/$sc/mpi1_$tag.log" | tr '\n' ' ')"
  for c in $CHUNKS; do
    $MPI "$dir/mpi2.py" --chunk "$c" > "$OUT/$sc/mpi2c${c}_$tag.log" 2>&1
    printf "  mpi2 c=%-4s %s\n" "$c" "$(grep -E 'Tiempo|LI=' "$OUT/$sc/mpi2c${c}_$tag.log" | tr '\n' ' ')"
  done
}

sweep() {
  sc="$1"; dir="$2"; mkdir -p "$OUT/$sc"
  echo "=== $sc warm-up ==="; round "$sc" "$dir" warmup
  for k in $(seq 1 "$RUNS"); do echo "=== $sc round $k ==="; round "$sc" "$dir" "run$k"; done
}

rm -rf /work && mkdir -p /work && cp "$APP_DIR"/*.py /work/ && cp -r "$APP_DIR/dataset" /work/dataset
sweep local /work
sweep bind "$APP_DIR"
echo Done
