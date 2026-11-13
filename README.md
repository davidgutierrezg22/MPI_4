# Parallel Word Counting in a Text Corpus with MPI

Lab 2 – Count how many times every word of `consulta.txt` appears in a corpus of
3,000 text files and report the 10 most frequent ones. The lab compares a sequential
baseline with two MPI implementations (`mpi4py`) that use different workload
distribution strategies.

---

## 1. Problem Description

`generator.py` (seed 42) creates `dataset/` with:

* `consulta.txt`: 200 distinct query words, one per line (all 200 occur in the corpus),
* `file_0001.txt` … `file_3000.txt`: 44,951,458 Spanish words in total (≈275 MB) drawn from a
  2,000-word vocabulary with Zipf-like weights.

File sizes are deliberately **heterogeneous** (`elegir_tamano_archivo`): 60 % of the
files have 2k–5k words, 30 % have 10k–20k words and 10 % have 50k–100k words. One large
file therefore holds as much work as 20–50 small ones. This is the main source of load
imbalance when files are split by count.

Expected output (it must match the sequential baseline):

```
Total de ocurrencias encontradas: 3631778
  a: 785774   para: 392156   sus: 228913   otros: 105530   ante: 99832
  unos: 88794 otra: 83901    vosotros: 61617 mios: 58420   tuya: 56635
```

---

## 2. Environment and Execution Instructions

| Item | Value |
|---|---|
| Host | Windows 11 Pro, AMD Ryzen 5 5500 (**6 physical cores / 12 hardware threads**), 16 GB RAM |
| Container runtime | Docker Desktop 29.9.2 (WSL2 backend) |
| Image | `augustosalazar/slim-mpi:2`: Python 3.8.20, mpi4py 4.1.1, Open MPI 4.1.4 |
| Process count | **p = 9**, the value recommended by `detectar_procesos_mpi.py` (see §3.3) |

### Repository layout

| Path | Content |
|---|---|
| `generator.py`, `baseline_secuencial.py`, `run_all.sh` | Provided by the course, **unchanged** |
| `mpi1.py` | MPI version 1: static distribution in contiguous blocks with the same number of files |
| `mpi2.py` | MPI version 2: dynamic self-scheduling with an atomic RMA counter (`--strategy lpt` gives a static size-aware alternative) |
| `tools/detectar_procesos_mpi.py` | Course script used to pick *p* (copied from the course repository) |
| `tools/run_experiments.sh` | Full experimental plan (warm-up + interleaved repeated runs) |
| `tools/run_chunk_sweep.sh` | Supplementary experiment: chunk size of `mpi2.py` |
| `tools/bench_listing.py` | Micro-benchmark: cost of listing the corpus (`isfile` vs `scandir`) |
| `tools/analyze.py`, `tools/analyze_chunks.py` | Compute T, S_p, E_p, LI and spread from the logs |
| `evidence/` | Raw logs and generated tables of all the runs reported here |

> `run_all.sh` deletes `results/` every time it runs, so the experiment evidence is kept in `evidence/`.

### Commands

Linux/macOS/Git Bash use `"$(pwd)":/app`. Windows `cmd` uses `"%cd%:/app"`. Run every command from this folder (`word_lab`).

```bash
# 0. Generate the dataset (once)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/generator.py

# 1. Choose p
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/tools/detectar_procesos_mpi.py

# 2. Single runs
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 python /app/baseline_secuencial.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 9 python /app/mpi1.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 9 python /app/mpi2.py
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 mpirun --allow-run-as-root --oversubscribe -np 9 python /app/mpi2.py --strategy lpt

# 3. Full experimental plan used in this report (~15 min), then the tables
docker run --rm -e RUNS=7 -e NATIVE_RUNS=5 -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/tools/run_experiments.sh
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/tools/run_chunk_sweep.sh
python tools/analyze.py          # -> evidence/summary.md, runs.csv, per_process.csv
python tools/analyze_chunks.py   # -> evidence/chunk_sweep/summary.md

# 4. Course batch script (np = 2 4 8, 3 runs each)
docker run --rm -v "$(pwd)":/app augustosalazar/slim-mpi:2 sh /app/run_all.sh
```

On Git Bash for Windows, prefix the commands with `MSYS_NO_PATHCONV=1` and use `"$(pwd -W)":/app`.

Each MPI program prints a per-process table (`rank, files, Mchars, tokens, local_time_s`),
machine-readable `PROC …`, `LI=` and `SPREAD=` lines, the top 10, and `EXECUTION_TIME=<s>`.
`run_all.sh` reads that last line. Each program also writes its full count to
`dataset/mpi1_results.csv` / `mpi2_results.csv` for comparison with `baseline_results.csv`.

---

## 3. Experimental Plan

### 3.1 Sequential baseline (what it does and how)

`baseline_secuencial.py` (provided) loads `consulta.txt` into a lower-cased `set`. It then iterates
over `os.listdir(dataset)`, keeps the files named `file_*.txt` that pass `os.path.isfile`, splits every line on whitespace,
lower-cases the tokens and increments a `Counter` for every token that is in the
query set. It times this whole loop with `time.perf_counter()`, without the CSV writing, prints `Tiempo de ejecución` and the top 10, and
saves `baseline_results.csv`. It provides **T_seq** and the reference output.

### 3.2 MPI version 1 (what it does and how)

`mpi1.py` follows the required scheme:

1. `Barrier`, then `t0 = MPI.Wtime()`, so all ranks start the clock together.
2. **rank 0 reads `consulta.txt`** and **broadcasts** the query set with `comm.bcast`.
3. **rank 0 lists the `file_*.txt` files** (sorted) and splits them **statically** into
   *p* contiguous blocks whose lengths differ by at most one file (3000/9 → 334 or 333 files), sent with
   `comm.scatter`.
4. Every rank runs **exactly the same counting loop as the baseline** on its block and measures
   its **local processing time** `t_i` around that loop only.
5. The partial `Counter`s, the number of files, characters, tokens and `t_i` are **gathered on rank 0**
   with `comm.gather`. Rank 0 merges them, stops the clock (same scope as the baseline timer: no CSV
   writing), and prints the per-process report, **LI**, spread and the **top 10**.

The only algorithmic change from the baseline is the file listing. It uses `os.scandir`, which
reads the file type from the directory entry, instead of one `os.path.isfile` call (a `stat`) per file.
Both produce the same 3,000 files. The effect of this change on the measured times is quantified in §4.1 and §5.3.

### 3.3 Test procedure

* **Correctness.** Every run's top 10 is compared with the baseline by `tools/analyze.py`, and the
  complete CSVs (all 200 words) were compared byte by byte with `cmp`. All runs matched.
* **Choice of p.** `detectar_procesos_mpi.py` (3 measurements per N = 1…12) recommended
  **N = 9** (median 1.673 s; N = 7 → 1.723 s, N = 11 → 1.708 s). Log: `evidence/detectar_procesos_mpi.log`.
  The single value p = 9 was then fixed for every experiment, as the lab requires.
* **Measurement protocol** (`tools/run_experiments.sh`):
  * the dataset is generated once and is identical for all runs;
  * one warm-up round of every program is executed and **discarded**, so all measured runs see a warm OS page cache;
  * **7 measured rounds** (≥ 3 required). Within a round the programs are **interleaved**
    (baseline → v1 → v2 → v2-LPT), so slow drifts in machine load affect all of them equally;
  * per run: total time `T` (internal timer), wall time of the `mpirun` command, and `t_i` of every rank;
  * metrics: `T_p` = mean of the 7 totals, `S_p = T_seq / T_p`, `E_p = S_p / p`,
    `LI = max(t_i)/mean(t_i)` and `spread = (max−min)/mean`, computed per run and averaged.
* **Two storage scenarios.**
  * **`bind`** (main): dataset read from the Windows folder mounted at `/app`, which is the official way to run the lab.
  * **`local`** (control, 5 rounds): the same programs after copying the project into the container's
    own filesystem. It removes the Windows→Linux file-sharing layer, whose metadata operations are very slow
    (≈1.4 ms per `stat`, §4.1).
* **Supplementary experiment.** `tools/run_chunk_sweep.sh` measures `mpi2.py` with chunk = 1, 4, 16, 64
  (5 interleaved rounds per scenario, with v1 as a reference).
* **Statistics.** Rounds are paired, so v1 and v2 are compared with paired differences per round
  (mean difference, standard deviation, t statistic, and the number of rounds each version won).
* Background applications (game client, chat, overlay, browser) were closed before the final runs.
  A first full run in which they were closed half-way through was discarded.

---

## 4. Experimental Plan Execution

### 4.1 Sequential baseline timing

| scenario | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 | run 7 | **T_seq** (mean ± sd) |
|---|---|---|---|---|---|---|---|---|
| bind  | 29.990 | 30.195 | 30.075 | 30.029 | 29.799 | 30.280 | 30.267 | **30.091 ± 0.172 s** |
| local | 10.210 | 9.697 | 9.796 | 9.791 | 10.021 | – | – | **9.903 ± 0.209 s** |

Output: 3,000 files, 44,951,458 tokens, 3,631,778 occurrences, and the top 10 listed in §1.

The baseline is **3× slower on the bind mount** than in the container's own filesystem. Part of this
is reading 275 MB through the file-sharing layer. Another part is the listing:
`tools/bench_listing.py` (`evidence/bench_listing.log`) measured **4.18–4.38 s** for
`listdir + isfile` over the 3,000 files on the bind mount, against 0.011 s for `scandir`.
In the local scenario both take less than 0.02 s.

### 4.2 MPI version 1 timing results (p = 9)

| scenario | run times (s) | **T_p** (mean ± sd) | mean wall (s) | **S_p** | **E_p** |
|---|---|---|---|---|---|
| bind  | 5.637, 5.725, 5.420, 5.214, 5.294, 5.823, 5.508 | **5.517 ± 0.225** | 6.034 | **5.45** | **0.606** |
| local | 2.314, 2.238, 2.168, 2.064, 2.535 | **2.264 ± 0.177** | 2.743 | **4.37** | **0.486** |

The top 10 matched the baseline in 12/12 runs. The wall time adds about 0.5 s of `mpirun` startup and interpreter
start, which neither program's internal timer includes.

### 4.3 Load imbalance evidence (per-process local times and LI)

Mean `t_i` per rank over the 7 runs (`bind`). The full per-run tables are in `evidence/summary.md`.

| rank | files | Mchars assigned | mean t_i (s) |
|---|---|---|---|
| 0 | 334 | **33.70** | **5.488** (slowest in 7/7 runs) |
| 1 | 334 | **26.70** | **4.702** (fastest in 7/7 runs) |
| 2 | 334 | 32.81 | 5.403 |
| 3 | 333 | 30.57 | 5.181 |
| 4 | 333 | 30.49 | 5.161 |
| 5 | 333 | 31.23 | 5.214 |
| 6 | 333 | 29.89 | 5.159 |
| 7 | 333 | 27.88 | 4.900 |
| 8 | 333 | 31.79 | 5.293 |
| **LI per run** | | | 1.063, 1.051, 1.061, 1.053, 1.076, 1.069, 1.063 → **mean 1.062** |
| **spread** | | | **0.152** (local scenario: LI **1.125**, spread 0.282) |

Evidence:

* Every rank received the same **number** of files (333–334), but the amount of **text** ranged
  from 26.7 to 33.7 M characters (+26 %). The correlation between characters assigned and mean `t_i` is
  **r = 0.991** (0.963 local). The imbalance comes from the data and is reproducible: the same ranks are
  slowest and fastest in every run.
* `T − max(t_i)` is only **0.029 s**, so the total time is set almost entirely by the slowest rank (rank 0).
  If the same work were perfectly balanced, every rank would finish at mean `t_i` = 5.167 s. The imbalance therefore costs
  ≈ 0.32 s (≈ 6 % of T_p) on the bind mount and ≈ 0.25 s (≈ 11 %) locally.

### 4.4 MPI version 2: how it corrects the imbalance, and its results

**Design.** Static splitting fails because the cost of a block is not known by counting files. The cost also depends on
how fast each rank runs: 9 ranks on 6 physical cores means some ranks share a core with another rank. `mpi2.py`
therefore lets the ranks pick up work as they become free:

* rank 0 reads and broadcasts the query (`bcast`) and broadcasts the sorted file list (`bcast`);
* one integer counter lives in an **MPI-3 RMA window** on rank 0 (`MPI.Win.Allocate`);
* every rank, **including rank 0** (no idle master), repeatedly calls the atomic
  `Fetch_and_op(+chunk, MPI.SUM)` to claim the next `chunk` files until the list is exhausted;
* local counting and the final `gather` and merge are identical to v1 (the code is imported from `mpi1.py`),
  so the **distribution policy is the only difference** between the two versions.

Ranks that finish sooner, because they got smaller files or a faster core, simply claim more chunks. The
only imbalance left is the tail of the work (at most one chunk).

**Choosing the chunk size.** A first version claimed one file at a time (chunk = 1). It removed the imbalance
(LI 1.008 on bind) but produced **no significant gain** on bind (6.109 s against 6.177 s for v1) and was
**7.8 % slower** locally (2.355 s against 2.185 s). The total work (Σ t_i) grew from 1.81× to 2.11× T_seq. The cause is 3,000
atomic operations: with Open MPI's passive-target RMA, the target (rank 0) serves a request only when
its Python loop enters an MPI call, so each claim can wait up to one file's processing time. Evidence:
`evidence/summary_v2_chunk1.md`. The chunk-size sweep (`evidence/chunk_sweep/summary.md`, 5 rounds each) gave:

| chunk | bind T (s) | bind LI | local T (s) | local LI |
|---|---|---|---|---|
| v1 (reference) | 5.488 | 1.064 | 2.321 | 1.114 |
| 1  | 5.567 | 1.010 | 2.445 | 1.016 |
| 4  | 5.440 | 1.012 | 2.343 | 1.014 |
| **16** | **5.266** | **1.011** | **2.304** | 1.017 |
| 64 | 5.550 | 1.082 | 2.334 | 1.097 |

Chunk = 64 leaves only ≈ 47 blocks for 9 ranks, so the imbalance at the end returns. Chunk = 16 (≈ 188 blocks,
≈ 21 per rank) keeps LI ≈ 1.01 with 16× fewer synchronisations, so **16 is the default**.

**Results of the delivered v2 (chunk = 16, p = 9).** These come from the same interleaved rounds as §4.1–4.3.

| scenario | run times (s) | **T_p** (mean ± sd) | mean wall (s) | **S_p** | **E_p** | **LI** | spread |
|---|---|---|---|---|---|---|---|
| bind  | 5.108, 5.173, 5.232, 5.242, 5.222, 4.993, 5.042 | **5.145 ± 0.099** | 5.687 | **5.85** | **0.650** | **1.008** | 0.024 |
| local | 1.991, 1.958, 2.116, 2.063, 1.993 | **2.024 ± 0.064** | 2.522 | **4.89** | **0.544** | **1.016** | 0.045 |

Mean `t_i` per rank (bind): 5.043–5.090 s. The files claimed per rank varied from 304 to 368 between runs, while the
characters per rank were 30.2–30.8 M. The top 10 matched the baseline in 12/12 runs.

**Comparison of the versions (p = 9).**

| | v1 static blocks | **v2 dynamic (chunk 16)** | v2-alt static LPT by size |
|---|---|---|---|
| T_p bind (s) | 5.517 ± 0.225 | **5.145 ± 0.099** | 5.947 ± 0.252 |
| S_p / E_p bind | 5.45 / 0.606 | **5.85 / 0.650** | 5.06 / 0.562 |
| LI / spread bind | 1.062 / 0.152 | **1.008 / 0.024** | 1.037 / 0.049 |
| T_p local (s) | 2.264 ± 0.177 | **2.024 ± 0.064** | 2.175 ± 0.067 |
| S_p / E_p local | 4.37 / 0.486 | **4.89 / 0.544** | 4.55 / 0.506 |
| LI local | 1.125 | **1.016** | 1.081 |
| paired v1 − v2 (bind) | | **−0.373 s (−6.8 %)**: v2 faster in 6/7 rounds, t = 3.23 (df = 6) | |
| paired v1 − v2 (local) | | **−0.240 s (−10.6 %)**: v2 faster in 5/5 rounds, t = 2.45 (df = 4) | |

v2 is also **more stable**: its standard deviation is less than half of v1's. Its slowest rank (5.113 s) finishes at
almost exactly the time v1 would reach with perfect balance (5.167 s). It recovers essentially all of the
≈ 0.32 s that v1 lost to imbalance.

The static **LPT** alternative (`--strategy lpt`) gives every rank exactly the same number of bytes
(30.56 M each), yet its LI is still 1.037 (bind) and 1.081 (local). Rank 0 is consistently the slowest
(5.32 s against ≈ 5.10 s). Equal data therefore does not mean equal time, because the ranks do not run at the same
speed. On the bind mount, LPT also pays **0.63 s** before the counting starts, to `stat` the 3,000 files
(even when the work is split among the ranks), and this makes it 7.8 % *slower* than v1.

### 4.5 Check with the course batch script (`run_all.sh`, np = 2, 4, 8)

`run_all.sh` ran unchanged and finished with exit code 0. It regenerated the identical dataset (seed 42), and all 19 logs
show the reference top 10. The baseline does not print `EXECUTION_TIME=`, so `times_summary.csv` contains only the MPI rows.
The single baseline run in that batch took 31.41 s. Logs: `evidence/run_all/`. The mean of 3 runs is shown below, with S_p computed against T_seq = 30.09 s from §4.1:

| np | v1 T (s) | v1 S_p | v1 mean LI | v2 T (s) | v2 S_p | v2 mean LI |
|---|---|---|---|---|---|---|
| 2 | 13.791 | 2.18 | 1.011 | **13.042** | 2.31 | 1.000 |
| 4 | 8.403 | 3.58 | 1.056 | **7.657** | 3.93 | 1.009 |
| 8 | 7.177 | 4.19 | 1.106 | **5.457** | 5.51 | 1.011 |

v2 was faster at every process count, and its LI stayed ≤ 1.011. At np = 2, S_p > 2 only because the baseline pays ≈ 4.25 s
for `isfile` on the bind mount (§4.1). These runs had no warm-up round and only 3 repetitions (one v1 np = 8 run had LI = 1.20),
so they are a consistency check, not the main measurement.

---

## 5. Analysis

### 5.1 Parallel performance

**Yes. MPI v1 clearly improved on the baseline:** 30.09 s → 5.52 s on the bind mount (**S₉ = 5.45, E₉ = 0.61**)
and 9.90 s → 2.26 s locally (**S₉ = 4.37, E₉ = 0.49**). That is about 50–60 % of the ideal S_p = 9.
Part of the bind speedup comes from the faster listing (≈ 4.25 s the baseline spends on `isfile`).
Taking that out (T_seq' = 25.84 s), v1 reaches S₉ ≈ 4.7 (E ≈ 0.52), which agrees with the local scenario, where listing
costs nothing.

The gap to S_p = p is **not** caused by serial code or communication. Everything outside the local
counting (reading the query, listing, `bcast`, `scatter`, `gather`, merge) takes **≈ 0.03 s on bind and 0.01 s locally**,
so Amdahl's serial fraction is below 1 %. The gap has two causes, and they multiply:

1. **Every rank works ≈ 1.8× slower than the sequential process.** The total local time Σ t_i is 1.82× T_seq
   locally and 1.80× the listing-free T_seq' on bind. The same instructions take longer per rank because:
   *p = 9 exceeds the 6 physical cores*, so at least 3 cores run two ranks on hyper-threads that share execution units and L1/L2 cache;
   all ranks share the 16 MB L3 cache, the memory bandwidth and the single file-sharing channel of
   Docker Desktop/WSL2; and the CPU boosts to a lower clock when every core is busy.
   With 6 cores plus SMT, the realistic upper limit is roughly 6–7.5, not 9.
2. **Load imbalance:** the total time is the *slowest* rank's time, which exceeds the mean by the factor LI.

These two factors explain the measured efficiency almost exactly: E ≈ 1/(1.8 × LI) predicts 0.49 for v1 local (measured 0.486)
and 0.55 for v2 local (measured 0.544).

### 5.2 Load balance

**Evidence in v1:** LI = **1.062** on bind and **1.125** locally, with a spread of 15 % and 28 %. The slowest rank
(rank 0, 33.7 M characters) and the fastest (rank 1, 26.7 M) were the same in every run. `t_i` follows the amount of
text assigned with r = 0.99. Equal numbers of files did not mean equal work, because 10 % of the files are 10–50× larger than the rest.
The imbalance is moderate rather than severe: each rank receives about 333 files, so the large files partly average out.
For that reason the possible gain is bounded at about 6 % (bind) and 11 % (local).

**v2 reduced it enough to give a real improvement.** LI fell to **1.008 / 1.016** and the spread to 2–5 %. T_p improved by
**6.8 % on bind** (faster in 6/7 paired rounds; mean paired gain 0.37 s vs. paired sd 0.31 s, t = 3.23, df = 6, p ≈ 0.02) and by **10.6 %
locally** (faster in 5/5 rounds). S₉ rose from 5.45 to **5.85** and E₉ from 0.61 to **0.65** (local: 4.37 → 4.89 and 0.49 → 0.54).
The gain matches what perfect balance could have recovered, so the remaining gap to ideal speedup is caused
by factor 1 in §5.1, not by imbalance. Two results show that a balancing scheme must also be cheap.
Chunk = 1 balanced just as well but gained nothing because of synchronisation costs. Size-aware LPT
balanced bytes perfectly but was still imbalanced in time, and its preparation cost made it slower than v1 on the bind mount.

### 5.3 Limitations

* **One machine, one p, small n.** All results come from a single 6-core desktop with p = 9 and 7 (bind) or 5 (local) rounds.
  The v1 vs v2 difference is consistent and statistically clear on bind (t = 3.2). Locally it is consistent
  (5/5 rounds) but n is small (t = 2.45, p ≈ 0.07). The *direction* of the conclusions is solid; the exact percentages are approximate to about ±3 %.
* **p was chosen with a CPU-bound prime-counting benchmark**, not with this partly I/O-bound workload, and its
  curve is flat and noisy between N = 7 and N = 11. A p ≤ 6 (physical cores) would probably give higher *efficiency*.
  We did not explore other values because the lab fixes a single p.
* **Storage layer.** On the official bind mount, I/O and `stat` calls are slow (≈ 1.4 ms per `stat`). This (a) makes the
  baseline pay ≈ 4.25 s for `isfile`, which inflates S_p on bind by about 16 % (5.45 → 4.7 without that cost), and (b) penalises any strategy that
  needs file metadata (LPT). We reduced this threat with the `local` control scenario, which leads to the same
  conclusions with smaller absolute speedups.
* **Warm page cache.** The 275 MB corpus fits in RAM, so after the discarded warm-up round every run reads from the OS page cache
  (warm-up 30.9 s vs. measured 30.1 s for the baseline). A truly cold read from disk was not measured; there the shared I/O
  channel would likely limit scaling even more.
* **Background noise.** Docker Desktop runs inside a WSL2 VM on a personal Windows PC. The background applications were closed, but OS noise
  remains (v1 sd ≈ 4 % of the mean). Interleaving the programs within each round and pairing the comparisons reduce, but do not remove, this effect.
* **Timer scope.** T excludes `mpirun` and interpreter startup (≈ 0.5 s, visible in the wall-time column) to match the baseline's
  timer. With wall time, S₉ is lower (v2 bind: 30.13/5.69 = 5.30) but the v1/v2 ranking is the same.
* **Fixed dataset.** One seed and one size distribution were used. With fewer or larger files per rank, static splitting would be
  more imbalanced and v2's advantage would grow; with uniform files, the gain would shrink to roughly the cost of synchronisation.

---

## 6. Conclusions

For this problem we recommend **dynamic self-scheduling with moderate chunks (MPI v2, chunk ≈ 16)**. Under identical conditions
with p = 9, it reached **S₉ = 5.85 and E₉ = 0.65** on the official setup (v1: 5.45 and 0.61) and **4.89 / 0.54** with storage
inside the container (v1: 4.37 / 0.49). It cut the load imbalance from **LI = 1.062 → 1.008** (1.125 → 1.016 locally), and it was faster
in 11 of the 12 paired rounds (−6.8 % and −10.6 %) with less than half of v1's run-to-run variance. Its advantage is that it adapts to *both*
sources of imbalance, heterogeneous file sizes and ranks running at different speeds on shared cores, without needing file
metadata. Static LPT, by contrast, balanced the bytes perfectly but still showed LI = 1.04–1.08 and lost 0.63 s to `stat` calls.
The chunk size matters: claiming one file at a time erased the benefit through synchronisation cost, and chunks that are too large brought the imbalance back.
Neither strategy comes close to the ideal speedup of 9. The remaining loss (E ≈ 0.55–0.65) is caused by running 9 ranks on 6 physical cores that share
cache, memory and I/O, not by communication or imbalance. Further gains would therefore come from matching p to the physical cores or
from faster local storage, not from a different distribution scheme.
