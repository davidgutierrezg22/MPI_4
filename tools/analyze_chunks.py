"""
Summarises evidence/chunk_sweep (tools/run_chunk_sweep.sh) into
evidence/chunk_sweep/summary.md.   Usage: python tools/analyze_chunks.py
"""
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "evidence", "chunk_sweep")


def main():
    lines = ["# mpi2.py chunk-size sweep (p=9, warm-up discarded)", ""]
    for sc in ("bind", "local"):
        path = os.path.join(D, sc)
        if not os.path.isdir(path):
            continue
        groups = {}
        for f in os.listdir(path):
            if "warmup" in f or not f.endswith(".log"):
                continue
            key = f.rsplit("_", 1)[0]
            txt = open(os.path.join(path, f), encoding="utf-8").read()
            t = float(re.search(r"EXECUTION_TIME=([0-9.]+)", txt).group(1))
            li = float(re.search(r"LI=([0-9.]+)", txt).group(1))
            groups.setdefault(key, []).append((t, li))
        keys = sorted(groups, key=lambda k: (k != "mpi1", int(k[5:]) if k.startswith("mpi2c") else 0))
        lines += [f"## Scenario `{sc}`", "",
                  "| program | runs | mean T (s) | std (s) | mean LI |", "|---|---|---|---|---|"]
        for k in keys:
            ts = [x[0] for x in groups[k]]
            label = "MPI v1 (static blocks)" if k == "mpi1" else f"MPI v2 dynamic, chunk={k[5:]}"
            lines.append(f"| {label} | {len(ts)} | {st.mean(ts):.3f} | {st.stdev(ts):.3f} | "
                         f"{st.mean(x[1] for x in groups[k]):.4f} |")
        lines.append("")
    out = os.path.join(D, "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
