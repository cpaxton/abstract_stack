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
