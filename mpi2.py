"""
MPI version 2 - load-balanced distribution of files.

Why v1 is imbalanced
--------------------
v1 gives every process the same NUMBER of files, but the generator produces
very heterogeneous files (60% small 2k-5k words, 30% medium 10k-20k, 10% large
50k-100k), so the work (tokens read) per process differs. On top of that, the
processes do not run at the same speed (hyper-threads sharing a core, OS and
Docker/WSL jitter), something no static plan can see in advance.

Default strategy: dynamic self-scheduling (``--strategy dynamic``)
------------------------------------------------------------------
- rank 0 reads consulta.txt and lists the files; both are broadcast (comm.bcast).
- A single integer counter lives in an MPI RMA window on rank 0.
- Every process, INCLUDING rank 0 (no idle master), repeatedly does an atomic
  ``Fetch_and_op(+chunk)`` on the counter to claim the next ``chunk`` files
  (default chunk=16, chosen with tools/run_chunk_sweep.sh: chunk=1 pays too much
  synchronisation, chunk=64 leaves too few blocks to balance the tail),
  until the counter passes the number of files.
- Fast processes simply claim more files, so the finishing times converge
  regardless of WHY a process is slow (bigger files or a slower core).
  The imbalance left is at most about one chunk of work.
- Partial counters are gathered in rank 0 exactly as in v1.

Alternative strategy: LPT by file size (``--strategy lpt``)
-----------------------------------------------------------
Static, size-aware: file sizes are measured in parallel (each rank stats 1/p
of the files, results gathered), then rank 0 applies Longest-Processing-Time
first (largest file -> currently least-loaded process) and scatters the lists.
It balances bytes, but cannot react to processes that run slower.

Run (inside the augustosalazar/slim-mpi:2 container):
    mpirun --allow-run-as-root --oversubscribe -np 9 python /app/mpi2.py
    mpirun --allow-run-as-root --oversubscribe -np 9 python /app/mpi2.py --strategy lpt
"""
import argparse
import heapq
import os
from collections import Counter

from mpi4py import MPI

# The query loading, file listing, counting loop and report are shared with v1
# so that the ONLY difference between both versions is the distribution policy.
from mpi1 import (CASE_SENSITIVE, CONSULTA_NAME, TOP_N, cargar_consulta,
                  contar_local, guardar_resultados_csv, listar_archivos, reportar)


def parse_args():
    ap = argparse.ArgumentParser(description="MPI word count, load-balanced version")
    ap.add_argument("--strategy", choices=["dynamic", "lpt"], default="dynamic")
    ap.add_argument("--chunk", type=int, default=16,
                    help="files claimed per atomic operation (dynamic strategy)")
    return ap.parse_args()


# ----------------------------------------------------------------- dynamic --
def contar_dinamico(comm, dataset_dir, archivos, palabras_objetivo, chunk):
    """Self-scheduling with an atomic shared counter (MPI-3 RMA)."""
    rank = comm.Get_rank()
    n = len(archivos)
    itemsize = MPI.LONG.Get_size()
    win = MPI.Win.Allocate(itemsize if rank == 0 else 0, itemsize, comm=comm)
    if rank == 0:
        win.Lock(0, MPI.LOCK_EXCLUSIVE)
        win.Put([bytearray(itemsize), MPI.LONG], 0)   # counter = 0
        win.Unlock(0)
    comm.Barrier()

    paso = bytearray(int(chunk).to_bytes(itemsize, "little", signed=True))
    previo = bytearray(itemsize)
    freq, tokens, chars, n_files = Counter(), 0, 0, 0

    tl0 = MPI.Wtime()
    win.Lock_all()
    while True:
        win.Fetch_and_op([paso, MPI.LONG], [previo, MPI.LONG], 0, 0, MPI.SUM)
        win.Flush(0)
        inicio = int.from_bytes(previo, "little", signed=True)
        if inicio >= n:
            break
        lote = archivos[inicio:inicio + chunk]
        f, t, c = contar_local(dataset_dir, lote, palabras_objetivo, CASE_SENSITIVE)
        freq.update(f)
        tokens += t
        chars += c
        n_files += len(lote)
    win.Unlock_all()
    local_time = MPI.Wtime() - tl0

    comm.Barrier()
    win.Free()
    return freq, tokens, chars, n_files, local_time


# --------------------------------------------------------------------- LPT --
def repartir_lpt(tamanos, archivos, p):
    """Greedy LPT: largest file first, always to the least-loaded process."""
    orden = sorted(zip(tamanos, archivos), key=lambda x: (-x[0], x[1]))
    heap = [(0, i) for i in range(p)]           # (bytes assigned, rank)
    bloques = [[] for _ in range(p)]
    for nbytes, fname in orden:
        carga, i = heapq.heappop(heap)
        bloques[i].append(fname)
        heapq.heappush(heap, (carga + nbytes, i))
    for b in bloques:
        b.sort()
    return bloques


def contar_lpt(comm, dataset_dir, archivos, palabras_objetivo):
    rank, size = comm.Get_rank(), comm.Get_size()
    # sizes measured in parallel: rank r stats files r, r+p, r+2p, ...
    mis_tam = [os.path.getsize(os.path.join(dataset_dir, f)) for f in archivos[rank::size]]
    todos = comm.gather(mis_tam, root=0)
    bloques = None
    if rank == 0:
        tamanos = [0] * len(archivos)
        for r, lst in enumerate(todos):
            tamanos[r::size] = lst
        bloques = repartir_lpt(tamanos, archivos, size)
    mis_archivos = comm.scatter(bloques, root=0)

    tl0 = MPI.Wtime()
    freq, tokens, chars = contar_local(dataset_dir, mis_archivos, palabras_objetivo,
                                       CASE_SENSITIVE)
    return freq, tokens, chars, len(mis_archivos), MPI.Wtime() - tl0


# -------------------------------------------------------------------- main --
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()
    t0 = MPI.Wtime()

    palabras_objetivo = None
    archivos = None
    if rank == 0:
        if not os.path.isdir(dataset_dir):
            print(f"Error: no existe la carpeta dataset en {dataset_dir}")
            comm.Abort(1)
        palabras_objetivo = cargar_consulta(os.path.join(dataset_dir, CONSULTA_NAME),
                                            CASE_SENSITIVE)
        archivos = listar_archivos(dataset_dir)
    palabras_objetivo = comm.bcast(palabras_objetivo, root=0)
    archivos = comm.bcast(archivos, root=0)

    if args.strategy == "dynamic":
        freq, tokens, chars, n_files, local_time = contar_dinamico(
            comm, dataset_dir, archivos, palabras_objetivo, max(1, args.chunk))
        nombre = f"MPI v2 (dynamic self-scheduling, chunk={max(1, args.chunk)})"
    else:
        freq, tokens, chars, n_files, local_time = contar_lpt(
            comm, dataset_dir, archivos, palabras_objetivo)
        nombre = "MPI v2 (static LPT by file size)"

    parte = {"rank": rank, "n_files": n_files, "chars": chars,
             "tokens": tokens, "local_time": local_time, "freq": freq}
    partes = comm.gather(parte, root=0)

    if rank == 0:
        freq_global = Counter()
        for pt in partes:
            freq_global.update(pt["freq"])
        freq_global.most_common(TOP_N)
        elapsed = MPI.Wtime() - t0

        out = "mpi2_results.csv" if args.strategy == "dynamic" else "mpi2_lpt_results.csv"
        resultados_path = os.path.join(dataset_dir, out)
        guardar_resultados_csv(resultados_path, freq_global)
        reportar(nombre, size, partes, freq_global, elapsed, dataset_dir, resultados_path)


if __name__ == "__main__":
    main()
