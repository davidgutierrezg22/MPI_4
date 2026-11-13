"""
Parses evidence/experiments/<scenario>/*.log produced by tools/run_experiments.sh
and computes, per scenario and program:
  T (mean, std, median, min, max) over the measured runs,
  S_p = T_seq / T_p,  E_p = S_p / p,
  LI = max(t_i)/mean(t_i) and spread = (max-min)/mean (mean over runs),
  mean local time of every rank over the runs.
Also checks that every run printed the same top 10 as the baseline.

Usage:  python tools/analyze.py            (from the word_lab folder)
Writes: evidence/summary.md, evidence/runs.csv, evidence/per_process.csv
"""
import csv
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(ROOT, "evidence", "experiments")
OUT = os.path.join(ROOT, "evidence")

PROGRAMS = [("baseline", "Sequential baseline"),
            ("mpi1", "MPI v1 - static blocks"),
            ("mpi2", "MPI v2 - dynamic self-scheduling"),
            ("mpi2lpt", "MPI v2 alt - static LPT by size")]

RE_TIME = re.compile(r"Tiempo de ejecuci.n: ([0-9.]+)")
RE_WALL = re.compile(r"WALL_TIME=([0-9.]+)")
RE_PROC = re.compile(r"PROC rank=(\d+) files=(\d+) chars=(\d+) tokens=(\d+) local_time=([0-9.]+)")
RE_TOP = re.compile(r"^  (\S+): (\d+)$", re.M)
RE_P = re.compile(r"processes p=(\d+)")


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = RE_TIME.search(txt)
    if not m:
        raise ValueError(f"no time in {path}")
    procs = [dict(rank=int(a), files=int(b), chars=int(c), tokens=int(d), t=float(e))
             for a, b, c, d, e in RE_PROC.findall(txt)]
    top = RE_TOP.findall(txt)
    mp = RE_P.search(txt)
    w = RE_WALL.search(txt)
    return {"T": float(m.group(1)), "wall": float(w.group(1)) if w else None,
            "procs": procs, "top": top, "p": int(mp.group(1)) if mp else 1}


def li_spread(procs):
    ts = [x["t"] for x in procs]
    mean = st.mean(ts)
    return max(ts) / mean, (max(ts) - min(ts)) / mean


def fmt(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def main():
    if not os.path.isdir(EXP):
        sys.exit(f"No experiments found in {EXP}")
    scenarios = sorted(d for d in os.listdir(EXP) if os.path.isdir(os.path.join(EXP, d)))
    md, runs_rows, proc_rows = [], [], []
    env = os.path.join(EXP, "environment.txt")
    if os.path.isfile(env):
        md += ["## Environment", "", "```", open(env, encoding="utf-8").read().strip(), "```", ""]

    for sc in scenarios:
        d = os.path.join(EXP, sc)
        data = {}
        for prog, _ in PROGRAMS:
            files = sorted(f for f in os.listdir(d) if re.fullmatch(rf"{prog}_run\d+\.log", f))
            data[prog] = [(f, parse(os.path.join(d, f))) for f in files]
        if not data["baseline"]:
            continue
        ref_top = data["baseline"][0][1]["top"]
        t_seq = st.mean(r["T"] for _, r in data["baseline"])

        md += [f"## Scenario `{sc}`", ""]
        md += ["### Per-run total times (s)", "",
               "| program | " + " | ".join(f"run {i+1}" for i in range(len(data['baseline']))) + " |",
               "|---|" + "---|" * len(data["baseline"])]
        for prog, label in PROGRAMS:
            if data[prog]:
                md.append(f"| {label} | " + " | ".join(fmt(r["T"]) for _, r in data[prog]) + " |")
        md.append("")

        md += ["### Summary", "",
               "| program | p | runs | mean T (s) | std (s) | median (s) | min | max | mean wall (s) | S_p | E_p | mean LI | mean spread | top-10 OK |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for prog, label in PROGRAMS:
            rs = [r for _, r in data[prog]]
            if not rs:
                continue
            Ts = [r["T"] for r in rs]
            p = rs[0]["p"]
            mean_T = st.mean(Ts)
            sd = st.stdev(Ts) if len(Ts) > 1 else 0.0
            S = t_seq / mean_T
            E = S / p
            ok = all(r["top"] == ref_top for r in rs)
            if prog == "baseline":
                li_m = sp_m = None
            else:
                lis = [li_spread(r["procs"]) for r in rs]
                li_m = st.mean(x[0] for x in lis)
                sp_m = st.mean(x[1] for x in lis)
            walls = [r["wall"] for r in rs if r["wall"] is not None]
            md.append(f"| {label} | {p} | {len(rs)} | {fmt(mean_T)} | {fmt(sd)} | {fmt(st.median(Ts))} | "
                      f"{fmt(min(Ts))} | {fmt(max(Ts))} | {fmt(st.mean(walls) if walls else None)} | "
                      f"{fmt(S, 2)} | {fmt(E, 3)} | {fmt(li_m, 4)} | {fmt(sp_m, 4)} | {'yes' if ok else 'NO'} |")
            for (fname, r) in data[prog]:
                li, sp = li_spread(r["procs"]) if r["procs"] else (None, None)
                ts = [x["t"] for x in r["procs"]]
                runs_rows.append([sc, prog, fname, p, r["T"], r["wall"], li, sp,
                                  max(ts) if ts else None, st.mean(ts) if ts else None,
                                  min(ts) if ts else None, r["top"] == ref_top])
                for x in r["procs"]:
                    proc_rows.append([sc, prog, fname, x["rank"], x["files"], x["chars"],
                                      x["tokens"], x["t"]])
        md.append("")

        for prog, label in PROGRAMS[1:]:
            rs = [r for _, r in data[prog]]
            if not rs:
                continue
            p = rs[0]["p"]
            md += [f"### Per-process local times - {label} (`{sc}`)", "",
                   "| rank | files | Mchars | " + " | ".join(f"t run {i+1}" for i in range(len(rs)))
                   + " | mean t_i |",
                   "|---|---|---|" + "---|" * (len(rs) + 1)]
            for rk in range(p):
                rows = [next(x for x in r["procs"] if x["rank"] == rk) for r in rs]
                files = sorted({x["files"] for x in rows})
                chars = st.mean(x["chars"] for x in rows) / 1e6
                files_s = str(files[0]) if len(files) == 1 else f"{min(files)}-{max(files)}"
                md.append(f"| {rk} | {files_s} | {chars:.2f} | "
                          + " | ".join(fmt(x["t"]) for x in rows)
                          + f" | {fmt(st.mean(x['t'] for x in rows))} |")
            lis = [li_spread(r["procs"]) for r in rs]
            md.append("| **LI** | | | " + " | ".join(fmt(x[0], 4) for x in lis)
                      + f" | **{fmt(st.mean(x[0] for x in lis), 4)}** |")
            md.append("| spread | | | " + " | ".join(fmt(x[1], 4) for x in lis)
                      + f" | {fmt(st.mean(x[1] for x in lis), 4)} |")
            md.append("")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Experiment summary (generated by tools/analyze.py)\n\n" + "\n".join(md) + "\n")
    with open(os.path.join(OUT, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "program", "log", "p", "T_s", "wall_s", "LI", "spread",
                    "t_max", "t_mean", "t_min", "top10_matches_baseline"])
        w.writerows(runs_rows)
    with open(os.path.join(OUT, "per_process.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "program", "log", "rank", "files", "chars", "tokens", "local_time_s"])
        w.writerows(proc_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
