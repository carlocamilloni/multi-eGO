#!/bin/bash
# Set up one bare (no backbone phi/psi term, no CMAP) MG-prior run of a capped peptide ACE-<seq>-NME.
#   usage: WORK=<work dir> bash setup_one.sh <seq one-letter, e.g. A or AW>
#   env:   WORK (default: current dir), PY (python with multi-eGO deps), GMX (gmx_mpi)
# Creates $WORK/inputs/ace_<seq>_nme (pdb2gmx) and $WORK/runs/ace_<seq>_nme (topol.tpr ready for run_one.sh).
set -euo pipefail
SEQ=$1
HERE=$(cd "$(dirname "$0")" && pwd)
MEGO=$(cd "$HERE/../.." && pwd)
WORK=$(cd "${WORK:-$PWD}" && pwd)
PY=${PY:-python}
GMX=${GMX:-gmx_mpi}
NAME=ace_${SEQ}_nme
export GMXLIB=$MEGO
IN=$WORK/inputs/$NAME
RUN=$WORK/runs/$NAME
mkdir -p "$IN" "$RUN"
cd "$IN"
$PY "$HERE/build.py" "$SEQ" "$NAME.pdb"
# termini: 'None' for both caps (ACE start, NME end)
printf "2\n2\n" | $GMX pdb2gmx -f "$NAME.pdb" -ff multi-ego-basic -water none -o conf.gro -p topol.top -i posre.itp -ter >pdb2gmx.log 2>&1
cd "$MEGO"
$PY multiego.py --system "$NAME" --egos mg --no_coil_cmap --inputs_dir "$WORK/inputs" --outputs_dir "$WORK/mego_out" \
    --explicit_name mg >"$RUN/multiego.log" 2>&1
OUT=$(ls -d "$WORK/mego_out/$NAME"/mg* | tail -1)
cd "$RUN"
cp "$OUT/ffnonbonded.itp" "$OUT/meGO.log" .
$PY "$HERE/strip_phipsi.py" "$OUT/topol_mego.top" topol_bare.top >strip.log
cp "$IN/posre.itp" .
$GMX editconf -f "$IN/conf.gro" -o start.gro -box 8 -c >editconf.log 2>&1
cp "$HERE/bare_300K.mdp" ff.mdp
$GMX grompp -f ff.mdp -c start.gro -p topol_bare.top -o topol.tpr -po mdout.mdp >grompp.out 2>&1
rm -f mdout.mdp
echo "$NAME: $(sed 's/.*removed: //' strip.log)"
