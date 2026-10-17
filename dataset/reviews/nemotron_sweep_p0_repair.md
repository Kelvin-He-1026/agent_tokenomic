# Nemotron sweep P0 repair

The source sweep has90 runs:15 selected questions ×3 frameworks ×2 thinking modes.
The frozen task definitions and source telemetry are preserved. The semantic
judge configuration is `claude-opus-5-5`, high effort, with validated normalized
score range0–1 and inclusive threshold0.8.

| Fix | Verified result |
| --- | --- |
| Semantic metric wiring | Legacy diagnostic-only metric lists now select the12 missing `semantic_correctness` judgments; explicit diagnostics-only opt-out remains available. |
| Output caps | All6 length terminations now set `output_cap_reached=true` and `per_call_output_cap_reached=true`; all90 total-cap flags remain false. |
| Termination |59 natural completions,7 forced finals,18 step limits,6 truncations. Strict success requires correct content, compliant format and natural completion for every framework. |
| Validation |87 offline tests pass, including actual sweep replay and live-runner cap recording. |

The12 semantic runs were rejudged after explicit user permission to send the
selected answers/reference context to Claude Code. All12 calls succeeded:
**1 passed and11 failed at threshold0.8**. The corrected CSV now has12 populated
semantic-score/threshold/version columns and **zero pending judgments**.
A dry-run with the old diagnostic metric list schedules zero remaining items.

The final strict-success count is **24/90**. All7 forced finals are excluded
from strict success while their substantive correctness remains visible.
All6 selected15-task cell plans are complete and fully judged; this is a
15-task pilot, not a completed100-task benchmark.25 deterministic answers
remain explicitly `unscorable_format`, with unknown content correctness and
no verified success credit. The threshold still needs human calibration;
these repairs verify numerical interpretation and pipeline consistency.
The original CSV, frozen task definitions and source run/call/tool/config/plan
files are unchanged. Historical eval rows are preserved byte-for-byte;12
new semantic verdicts were appended.

Artifacts:

- [Corrected CSV](../../output/runs/thinking_sweep_NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16_20261016_120206/results_p0_fixed_20261016_141746.csv)
- [Quality report](../../output/runs/thinking_sweep_NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16_20261016_120206/results_p0_fixed_20261016_141746.quality.json)
- [Repair manifest, hashes and pending run IDs](nemotron_sweep_p0_repair.json)

Rejudging is complete. The manifest retains the exact command, approval
history, normalized verdicts, model/CLI provenance, input hashes and output
paths. Re-running the same command resumes without duplicate judge calls.

- [Judge execution log](../../output/judge_nemotron_p0_semantic_20261016_142343.log)
- [Judge call accounting](../../output/judge_calls_20261016_142345.jsonl)
