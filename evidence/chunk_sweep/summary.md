# mpi2.py chunk-size sweep (p=9, warm-up discarded)

## Scenario `bind`

| program | runs | mean T (s) | std (s) | mean LI |
|---|---|---|---|---|
| MPI v1 (static blocks) | 5 | 5.488 | 0.227 | 1.0641 |
| MPI v2 dynamic, chunk=1 | 5 | 5.567 | 0.165 | 1.0095 |
| MPI v2 dynamic, chunk=4 | 5 | 5.440 | 0.082 | 1.0120 |
| MPI v2 dynamic, chunk=16 | 5 | 5.266 | 0.197 | 1.0108 |
| MPI v2 dynamic, chunk=64 | 5 | 5.550 | 0.202 | 1.0820 |

## Scenario `local`

| program | runs | mean T (s) | std (s) | mean LI |
|---|---|---|---|---|
| MPI v1 (static blocks) | 5 | 2.321 | 0.156 | 1.1135 |
| MPI v2 dynamic, chunk=1 | 5 | 2.445 | 0.079 | 1.0162 |
| MPI v2 dynamic, chunk=4 | 5 | 2.343 | 0.219 | 1.0143 |
| MPI v2 dynamic, chunk=16 | 5 | 2.304 | 0.371 | 1.0167 |
| MPI v2 dynamic, chunk=64 | 5 | 2.334 | 0.157 | 1.0972 |

