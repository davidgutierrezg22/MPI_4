"""
MPI version 1 - static block distribution of files.

- rank 0 reads consulta.txt and broadcasts the query words (comm.bcast)
- rank 0 lists the file_*.txt files and splits them into p contiguous
  blocks of (almost) the same number of files (static distribution, comm.scatter)
- every process counts the query words in its own files
- partial Counters are gathered in rank 0 (comm.gather), merged, and the
  top 10 is printed (must match baseline_secuencial.py)

Run (inside the augustosalazar/slim-mpi:2 container):
    mpirun --allow-run-as-root --oversubscribe -np 4 python /app/mpi1.py
"""
import os
import sys
import time
from collections import Counter

from mpi4py import MPI

CONSULTA_NAME = "consulta.txt"
CASE_SENSITIVE = False
TOP_N = 10
OUTPUT_FILE = "mpi1_results.csv"


def cargar_consulta(consulta_path, case_sensitive=False):
    """Same semantics as the baseline: one word per line, lower-cased set."""
    with open(consulta_path, "r", encoding="utf-8") as f:
        palabras = [line.strip() for line in f if line.strip()]
    if not case_sensitive:
        palabras = [w.lower() for w in palabras]
    return set(palabras)


def listar_archivos(dataset_dir):
    """
    Sorted list of file_*.txt (same filter as the baseline).
    os.scandir gets the file type from the directory entry itself, so it does
    not need one stat() per file like os.path.isfile (on a Docker bind mount
    from Windows each stat() costs ~1.5 ms -> ~4.4 s for 3000 files).
    """
    archivos = [e.name for e in os.scandir(dataset_dir)
                if e.name.startswith("file_") and e.name.endswith(".txt") and e.is_file()]
    archivos.sort()
    return archivos


def repartir_bloques(archivos, p):
    """Static distribution: p contiguous blocks whose sizes differ by at most 1."""
    n = len(archivos)
    base, resto = divmod(n, p)
    bloques, inicio = [], 0
    for i in range(p):
        fin = inicio + base + (1 if i < resto else 0)
        bloques.append(archivos[inicio:fin])
        inicio = fin
    return bloques


def contar_local(dataset_dir, archivos, palabras_objetivo, case_sensitive=False):
    """Exactly the same counting loop as the baseline, on a subset of files."""
    freq = Counter()
    tokens = 0
    chars = 0
    for fname in archivos:
        ruta = os.path.join(dataset_dir, fname)
        with open(ruta, "r", encoding="utf-8") as f:
            for linea in f:
                chars += len(linea)
                palabras = linea.split()
                if not case_sensitive:
                    palabras = [w.lower() for w in palabras]
                tokens += len(palabras)
                for w in palabras:
                    if w in palabras_objetivo:
                        freq[w] += 1
    return freq, tokens, chars


def guardar_resultados_csv(path, freq):
    with open(path, "w", encoding="utf-8") as f:
        f.write("palabra,conteo\n")
        for palabra in sorted(freq):
            f.write(f"{palabra},{freq[palabra]}\n")


def reportar(nombre, size, partes, freq_global, elapsed, dataset_dir, resultados_path):
    """Prints per-process stats, load-imbalance metrics and the top 10 (rank 0 only)."""
    tiempos = [pt["local_time"] for pt in partes]
    media = sum(tiempos) / len(tiempos)
    li = max(tiempos) / media if media > 0 else float("nan")
    spread = (max(tiempos) - min(tiempos)) / media if media > 0 else float("nan")

    print(f"=== {nombre} | processes p={size} ===")
    print(f"Tiempo de ejecución: {elapsed:.6f} segundos\n")
    print(f"Dataset procesado: {dataset_dir}")
    print(f"Archivos procesados: {sum(pt['n_files'] for pt in partes)}")
    print(f"Total de tokens leídos: {sum(pt['tokens'] for pt in partes)}")
    print(f"Total de ocurrencias encontradas: {sum(freq_global.values())}")
    print(f"Resultados guardados en: {resultados_path}\n")

    print("Per-process report:")
    print(f"  {'rank':>4} {'files':>6} {'Mchars':>8} {'tokens':>10} {'local_time_s':>13}")
    for pt in partes:
        print(f"  {pt['rank']:>4} {pt['n_files']:>6} {pt['chars'] / 1e6:>8.2f} "
              f"{pt['tokens']:>10} {pt['local_time']:>13.6f}")
    print()
    for pt in partes:
        # machine-readable lines (parsed by tools/analyze.py)
        print(f"PROC rank={pt['rank']} files={pt['n_files']} chars={pt['chars']} "
              f"tokens={pt['tokens']} local_time={pt['local_time']:.6f}")
    print(f"LOCAL_MEAN={media:.6f}")
    print(f"LOCAL_MAX={max(tiempos):.6f}")
    print(f"LOCAL_MIN={min(tiempos):.6f}")
    print(f"LI={li:.4f}")
    print(f"SPREAD={spread:.4f}\n")

    print(f"Top {TOP_N} palabras de consulta en el corpus:")
    for palabra, cuenta in freq_global.most_common(TOP_N):
        print(f"  {palabra}: {cuenta}")
    print()
    print(f"EXECUTION_TIME={elapsed:.6f}")
    sys.stdout.flush()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")

    comm.Barrier()                 # every process starts the clock together
    t0 = MPI.Wtime()

    # 1) rank 0 reads the query and broadcasts it
    palabras_objetivo = None
    bloques = None
    if rank == 0:
        if not os.path.isdir(dataset_dir):
            print(f"Error: no existe la carpeta dataset en {dataset_dir}")
            comm.Abort(1)
        palabras_objetivo = cargar_consulta(os.path.join(dataset_dir, CONSULTA_NAME),
                                            CASE_SENSITIVE)
        # 2) rank 0 lists the files and builds the static distribution
        bloques = repartir_bloques(listar_archivos(dataset_dir), size)
    palabras_objetivo = comm.bcast(palabras_objetivo, root=0)
    mis_archivos = comm.scatter(bloques, root=0)

    # 3) local work
    tl0 = MPI.Wtime()
    freq, tokens, nchars = contar_local(dataset_dir, mis_archivos, palabras_objetivo,
                                        CASE_SENSITIVE)
    local_time = MPI.Wtime() - tl0

    # 4) gather partial results in rank 0
    parte = {"rank": rank, "n_files": len(mis_archivos), "chars": nchars,
             "tokens": tokens, "local_time": local_time, "freq": freq}
    partes = comm.gather(parte, root=0)

    if rank == 0:
        freq_global = Counter()
        for pt in partes:
            freq_global.update(pt["freq"])
        freq_global.most_common(TOP_N)
        elapsed = MPI.Wtime() - t0     # same scope as the baseline timer (no CSV writing)

        resultados_path = os.path.join(dataset_dir, OUTPUT_FILE)
        guardar_resultados_csv(resultados_path, freq_global)
        reportar("MPI v1 (static blocks by file count)", size, partes, freq_global,
                 elapsed, dataset_dir, resultados_path)


if __name__ == "__main__":
    main()
