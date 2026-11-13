"""Cost of listing the corpus: baseline style (listdir + isfile per file) vs
os.scandir (used by mpi1/mpi2). Usage: python tools/bench_listing.py <dataset_dir> [reps]"""
import os, sys, time
d = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for k in range(reps):
    t = time.perf_counter()
    a = [f for f in os.listdir(d) if f.startswith("file_") and f.endswith(".txt")
         and os.path.isfile(os.path.join(d, f))]
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    b = [e.name for e in os.scandir(d) if e.name.startswith("file_") and e.name.endswith(".txt") and e.is_file()]
    t2 = time.perf_counter() - t
    assert sorted(a) == sorted(b)
    print(f"{d} rep{k+1}: listdir+isfile={t1:.3f}s  scandir={t2:.3f}s  files={len(a)}")
