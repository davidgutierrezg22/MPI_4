#!/bin/sh
# Experimental plan for Lab 2 (run inside augustosalazar/slim-mpi:2):
#
#   docker run --rm -v "%cd%:/app" augustosalazar/slim-mpi:2 sh /app/tools/run_experiments.sh
#   docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/tools/run_experiments.sh
#
# Assumes dataset/ was already generated with generator.py (not regenerated,
# so every run reads exactly the same corpus).
#
# Design
#   * p is fixed to the value recommended by tools/detectar_procesos_mpi.py (NP, default 9).
#   * 1 warm-up round (discarded) so that every measured run sees a warm page cache.
#   * RUNS measured rounds (default 5 >= 3 required). Inside each round the
#     programs are interleaved (baseline, mpi1, mpi2, mpi2-lpt) so that slow
#     drifts of the machine load affect all programs equally.
#   * Scenario "bind": dataset read from the Windows folder mounted in /app
#     (the official way of running the lab).
#   * Scenario "local": same programs, but the project is first copied to the
#     container's own filesystem (/work) to isolate the cost of the
#     Windows->Linux bind mount. (NATIVE_RUNS rounds, set 0 to skip.)
#
# Output: evidence/experiments/<scenario>/<program>_run<k>.log
set -eu

APP_DIR="/app"
NP="${NP:-9}"
RUNS="${RUNS:-5}"
NATIVE_RUNS="${NATIVE_RUNS:-3}"
OUT="$APP_DIR/evidence/experiments"
MPI="mpirun --allow-run-as-root --oversubscribe -np $NP python"

rm -rf "$OUT"
mkdir -p "$OUT"
{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nproc=$(nproc)"
  echo "np=$NP runs=$RUNS native_runs=$NATIVE_RUNS"
  python --version 2>&1
  python -c "import mpi4py; print('mpi4py', mpi4py.__version__)"
  mpirun --version | head -n 1
  grep -m1 "model name" /proc/cpuinfo || true
  grep MemTotal /proc/meminfo || true
} > "$OUT/environment.txt"

now() { date +%s.%N; }

run_one() {   # scenario dir program label log
  dir="$1"; prog="$2"; log="$3"
  t0=$(now)
  case "$prog" in
    baseline) python "$dir/baseline_secuencial.py" > "$log" 2>&1 ;;
    mpi1)     $MPI "$dir/mpi1.py" > "$log" 2>&1 ;;
    mpi2)     $MPI "$dir/mpi2.py" > "$log" 2>&1 ;;
    mpi2lpt)  $MPI "$dir/mpi2.py" --strategy lpt > "$log" 2>&1 ;;
  esac
  t1=$(now)
  echo "WALL_TIME=$(echo "$t1 $t0" | awk '{printf "%.6f", $1-$2}')" >> "$log"
  grep -E "Tiempo de ejecuci|LI=" "$log" | tr '\n' ' '
  echo
}

scenario() {  # name dir runs
  name="$1"; dir="$2"; runs="$3"
  mkdir -p "$OUT/$name"
  echo "=== scenario $name ($dir), warm-up ==="
  for prog in baseline mpi1 mpi2 mpi2lpt; do
    printf "  warmup %-9s " "$prog"; run_one "$dir" "$prog" "$OUT/$name/${prog}_warmup.log"
  done
  for k in $(seq 1 "$runs"); do
    echo "=== scenario $name, round $k/$runs ==="
    for prog in baseline mpi1 mpi2 mpi2lpt; do
      printf "  %-9s " "$prog"; run_one "$dir" "$prog" "$OUT/$name/${prog}_run${k}.log"
    done
  done
}

scenario bind "$APP_DIR" "$RUNS"

if [ "$NATIVE_RUNS" -gt 0 ]; then
  echo "=== copying project to container filesystem (/work) ==="
  rm -rf /work && mkdir -p /work
  cp "$APP_DIR"/*.py /work/
  cp -r "$APP_DIR/dataset" /work/dataset
  scenario local /work "$NATIVE_RUNS"
fi

echo "Done. Logs in $OUT"
