# Shelf-stocking research draft

This is a working problem statement and experiment plan, not a submission or a
completed-results paper. It follows `home_robot_v3/paper`'s organization: a root
LaTeX file, numbered section files, bibliography, build script and ignored build
artifacts. It uses the standard `article` class until we choose a venue; it does
not copy an unrelated manuscript or assume a CoRL submission.

```bash
./paper/build.sh
./paper/build.sh --clean
```

Build requires `latexmk`, `pdflatex` and BibTeX with the ordinary TeX Live LaTeX
packages (`texlive-latex-extra`, `texlive-fonts-recommended`,
`texlive-bibtex-extra` on Ubuntu). The simulator installation does not install
TeX or download container images. The PDF is generated locally as `main.pdf`.

- [main.tex](main.tex): document root and draft status.
- [sections/](sections/): motivation, related work, physical task, controller and
  task-policy proposal, planned evaluation, current status, reproducibility.
- [references.bib](references.bib): verified source metadata; pin reviewed versions.
- [reviews/agentic_gentamp.md](reviews/agentic_gentamp.md): paper review and our
  implementation priorities, with primary-source links.
- [MABEL controller](../docs/mabel_controller.md): tracking API, diagnostics,
  limits, tests, and proposed RL integration contract.
- [Contact-control report](../docs/reports/contact_control_followup.md): public
  regression and fresh-scene measurements, policy revisions, stage counts,
  rejected experiments, native-crash censoring and dual-view evidence.
- [Thumb-preshape follow-up](../docs/reports/thumb_preshape_followup.md): native
  thumb clearance, complete-task regression/fresh-scene evaluation and retained
  compliant-control search results.
- [Selected compliance study](../docs/reports/compliance_followup.md): selected
  force targets, forty matched scene pairs and explicit ample-space boundaries.

Keep observed diagnostics, literature claims and proposed experiments distinct.
Every result needs a configuration, revision, seed suite and saved evidence.
Held-out seeds/episodes must remain outside agent-visible development files.
Do not claim successful stocking, bimanual coordination, recovery or MABEL RL
integration from an interface, scene render, controller unit test or paper build.
