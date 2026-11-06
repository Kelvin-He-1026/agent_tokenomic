# GAIA load-test level summaries

One row per concurrency level (1, 4, 16, 32, 64 virtual users) for each sweep,
as written by `scripts/run_load.sh` (`<run>_levels.csv`). These are aggregates
only: no GAIA questions, answers or transcripts, which stay in the ignored
`output/` tree (GAIA asks that they not be reshared).

Common setup: Qwen3.8-27B BF16, TP=2 on 2x RTX PRO 6000, thinking on at low
effort; 24 GAIA questions (6 excluded); tool mode record; `--no-stop`. Host:
2x Xeon 6740P (48 cores per socket, SMT on, 192 logical CPUs).

| File | Sweep | CPUs for the test | Sandbox thread limit (RLIMIT_NPROC) |
| --- | --- | --- | --- |
| `load_v2_langgraph_levels.csv` | LangGraph ReAct (in-process) | all | n/a |
| `keenable_langgraph_levels.csv` | LangGraph with Keenable search, 1 user | all | n/a |
| `load_v2_512t_hermes_levels.csv` | NemoClaw Hermes | all (192) | 512 (NemoClaw default) |
| `load_32c_1024t_hermes_c1_levels.jsonl` + `load_32c_1024t_hermes_c4-64_levels.csv` | NemoClaw Hermes; 1 user before a power cut, 4-64 resumed | 32 cores/socket, no SMT (64) | 1024 |
| `load_48c_1024t_hermes_levels.csv` | NemoClaw Hermes | all (192) | 1024 |
| `load_16c_2048t_hermes_levels.csv` | NemoClaw Hermes | 16 cores/socket, no SMT (32) | 2048 |
| `load_8c_2048t_hermes_levels.csv` | NemoClaw Hermes | 8 cores/socket, no SMT (16) | 2048 |

## NemoClaw Hermes at a glance

Successful tasks/min (infra error rate):

| Users | 512, all CPUs | 32c-1024t | 48c-1024t | 16c-2048t | 8c-2048t |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.060 | 0.094 | 0.087 | 0.085 | 0.083 |
| 4 | 0.41 | 0.40 | 0.38 | 0.29 | 0.39 |
| 16 | 0.71 (10%) | 1.02 (0%) | 0.79 (8%) | 0.80 (4%) | 0.85 (6%) |
| 32 | 1.16 (18%) | 1.26 (11%) | 1.26 (14%) | 1.22 (10%) | 1.33 (8%) |
| 64 | 2.22 (42%) | 1.50 (31%) | 1.49 (39%) | 1.34 (31%) | 1.64 (26%) |

- At 512 and 1024 the sandbox user hit RLIMIT_NPROC at 32-64 users (thread
  pools sized to the visible CPUs failed with EAGAIN). At 2048 there were no
  thread failures (peak 985 sandbox-user threads at 16c, 912 at 8c). The 512
  run's high 64-user throughput comes from crashed runs freeing the GPU early.
- At 64 users the GPU is the limit (vLLM queue and KV cache near full); the
  remaining infra errors are runs past the 3600 s deadline.
- The pinned CPUs were never the bottleneck: mean busy 6% (32c), 12% (16c),
  24% (8c).
