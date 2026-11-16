# Development evidence

`development_runs.json` extracts selected actual solver artifacts, including an
unverified native rendering failure. It is not a held-out experiment or the
complete controller-tuning history and must not be used to calculate a benchmark
success rate. Full controls, states, force traces and MP4s remain in ignored
`outputs/` directories.

Regenerate from the local artifacts with:

```bash
.venv/bin/python scripts/summarize_stocking_runs.py \
  outputs/random-cartesian-two0 outputs/random-cartesian-two1 \
  outputs/random-two-seed0-video outputs/random-success-seed1 \
  --output paper/evidence/development_runs.json
```

Each entry retains the seed, model/software versions, physical gates, measured
outcome and artifact source. Unknown/crashed episodes remain unknown rather than
being silently removed or treated as successful video generation.

`thumb_approach_comparison.csv` contains measured first-approach/closure evidence
for development scene 112, with and without the 0.4 rad thumb MCP preshape.
Each 20 ms bin preserves the maximum combined arm/hand carton-normal force from
its original 2 ms measurements. Position is the final sample in the bin. The
neutral-thumb run aborts at the fixed 25 N limit; the preshaped run completes
both cartons. The plot covers approach/closure only, not the complete episode.

Regenerate from full native artifacts, or omit the two run arguments to plot
the committed CSV alone:

```bash
.venv/bin/python scripts/plot_grasp_comparison.py \
  --baseline outputs/thumb-ablation-dev-v1/seed-112 \
  --selected outputs/thumb-preshape-dev-v1/seed-112
```

The compact multi-campaign report lives in
`docs/reports/thumb_preshape_followup.json`; its accompanying Markdown includes
commands, policy revisions, rejected probes and the complete search budget.
