#!/bin/bash
# Run (or continue) one bare capped-peptide run and extract phi/psi with gmx chi.
#   usage: run_one.sh <runs/ace_X_nme dir>      env: GMX (gmx_mpi), MAXH, KEEP_XTC (1)
set -euo pipefail
GMX=${GMX:-gmx_mpi}
cd "$1"
[[ -f done ]] && exit 0
args=(-s topol.tpr -ntomp 1 -cpt 15)
[[ -f state.cpt ]] && args+=(-cpi state.cpt)
[[ -n "${MAXH:-}" ]] && args+=(-maxh "$MAXH")
$GMX mdrun "${args[@]}" > mdrun.out 2>&1
if [[ -f confout.gro ]]; then      # written only when nsteps is reached
    mkdir -p dih && (cd dih && echo 0 | $GMX chi -s ../topol.tpr -f ../traj_comp.xtc -all -phi -psi \
        -o order.xvg -g chi.log > chi.out 2>&1)
    [[ "${KEEP_XTC:-1}" == 0 ]] && rm -f traj_comp.xtc
    touch done
fi
