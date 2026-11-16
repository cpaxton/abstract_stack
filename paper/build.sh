#!/usr/bin/env bash
# Build the research draft with a locally installed TeX toolchain.
set -euo pipefail
paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--clean" ) ]]; then
    echo "Usage: $0 [--clean]" >&2
    exit 2
fi
if ! command -v latexmk >/dev/null 2>&1; then
    echo "Install latexmk and TeX Live (latex-extra, fonts-recommended, bibtex-extra)." >&2
    exit 1
fi
if [[ "${1:-}" == "--clean" ]]; then
    latexmk -C main.tex
else
    latexmk -pdf -halt-on-error -interaction=nonstopmode -file-line-error main.tex
fi
