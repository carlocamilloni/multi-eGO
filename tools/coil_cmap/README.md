# coil_cmap — the transferable coil CMAP of multi-eGO

multi-ego-basic.ff has no backbone φ/ψ cosines: each residue carries one zero-strength placeholder per backbone torsion (`gd_cmap`). multi-eGO (`src/multiego/cmap.py`) then writes, for both `--egos mg` and `--egos production`:
- **`coil_cmap.itp`:** a `[ cmaptypes ]` file, included after `ffnonbonded.itp`;
- **a `[ cmap ]` section** for every protein residue that has both backbone neighbours.

The grids come from `src/multiego/data/coil_cmap_300K.npz`. This folder holds the data and scripts that produced them.

## The library

    V(φ,ψ | L,X,R) = −kT ln p_coil(L,X,R) + kT ln p_bare(L,X,R),   T_fit = 300 K
    ln p_coil(L,X,R) = ln p(X|L) + ln p(X|R) − ln p(X|ALL)                         (NDRD TCB, Ting et al. 2010)
    ln p_bare(L,X,R) = ln p(ACE-L-X-NME: X) + ln p(ACE-X-R-NME: X) − ln p(ACE-X-NME: X)

- **Contents:** 20 residue types × 21 left × 21 right neighbours (the 20 types plus `CAP` for an ACE/NME/NH2 cap) = 8,820 grids of 24×24 points, in kJ/mol, φ outer loop, starting at −180°.
- **Fit settings:**
  - smoothing σ 7.5°;
  - density floor 0.01 × uniform;
  - zero mean under the coil target;
  - clipped at ±6 kT;
  - 5° cells averaged onto the GROMACS grid corners.
- **Coverage:** HIS is the single neutral form of the force field, CYS covers CYS2, and Pro is trans only.
- **Not covered by a CMAP:** ψ of a chain's first residue, φ of its last, and residues without a grid. These keep the original per-residue rtp cosines (`src/multiego/data/backbone_terminal_cosines.json`).
- **Old-style input topologies:** backbone φ/ψ dihedrals in topologies made with the previous force field are replaced by the CMAP. `--no_coil_cmap` keeps the input dihedrals unchanged.

**Bare references** (`data/bare_300K.npz`, uint32 φ/ψ counts on 5° cells; produced on 26 Nov 2026):
- Systems: 20 ACE-X-NME and 400 ACE-X-Y-NME.
- Force field: multi-ego-basic.ff at commit 5a3e97e8 (ideal bonded geometry), multi-eGO `--egos mg`, with every backbone φ/ψ dihedral removed (`strip_phipsi.py`).
- Box: 8 nm cubic.
- `bare_300K.mdp`: sd, dt 5 fs, tau_t 25 ps, 300 K, cut-off 2.5 nm, 200 ns per run, frames every 2.5 ps. The first 1 ns is dropped, leaving 79,600 frames per run.
- Convergence: half-run JSD has a median of 0.003 bits (max 0.006).

**Check.** Reweighting the ideal-geometry AB42 MG prior at 278 K from the previous AB42-specific NDRD CMAP onto this library moves the mean basins αR/β/PPII/αL from 42/25/19/14 to 35/30/23/12 (NDRD target 31/30/26/13), and the mean JSD to the target from 0.031 to 0.019 bits.

## Reproducing it

    PY=<python with the multi-eGO dependencies and PeptideBuilder>
    export WORK=<work dir>
    xargs -P 8 -n 1 env WORK=$WORK PY=$PY bash setup_one.sh < sequences.txt   # 420 systems, about 8 min on 8 cores
    NP=4 WORK=$WORK bash run_all.sh                                          # about 21 core-hours; writes runs/*/dih
    $PY fit_coil_cmap.py bare --work $WORK --out data/bare_300K.npz
    $PY fit_coil_cmap.py fit  --bare data/bare_300K.npz --lib <NDRD_TCB.txt> --out ../../src/multiego/data/coil_cmap_300K.npz
    $PY fit_coil_cmap.py check --library ../../src/multiego/data/coil_cmap_300K.npz --lib <NDRD_TCB.txt> \
        --top <topology of a run> --dih <run>/dih --temp <K> [--old-grids <npz of the run's current CMAP>]

- `build.py`: builds G-X(-Y)-G with PeptideBuilder and cuts the glycines to ACE/NME.
- `coil_lib.py`: the NDRD reader and grid utilities.
- The NDRD library file itself is not included. Download it from https://dunbrack.fccc.edu/lab/ndrd under its licence.

Coil target: D. Ting, G. Wang, M. Shapovalov, R. Mitra, M. I. Jordan, R. L. Dunbrack Jr., *Neighbor-dependent Ramachandran probability distributions of amino acids developed from a hierarchical Dirichlet process model*, PLoS Comput Biol 6:e1000763 (2010).
