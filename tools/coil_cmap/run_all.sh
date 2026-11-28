#!/bin/bash
# Run all 420 bare capped-peptide runs, NP at a time (1 core each).
#   WORK=<work dir> bash run_all.sh          # NP=8 by default
#   NP=32 MAXH=23.5 bash run_all.sh  # on a node with 32 cores; resubmit the same command to continue
#   GMX=gmx KEEP_XTC=0 bash run_all.sh
# Unfinished runs restart from their checkpoint; finished ones (marker file 'done') are skipped.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "${WORK:-$PWD}" && pwd)
NP=${NP:-8}
ls -d "$WORK"/runs/ace_*_nme | xargs -P "$NP" -n 1 bash "$HERE/run_one.sh"
echo "finished: $(ls "$WORK"/runs/*/done 2>/dev/null | wc -l) / $(ls -d "$WORK"/runs/ace_*_nme | wc -l)"
