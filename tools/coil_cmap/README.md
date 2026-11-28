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

**Bare references** (`data/bare_300K.npz`, uint32 φ/ψ counts on 5° cells; produced on 28 Nov 2026):
- Systems: 20 ACE-X-NME and 400 ACE-X-Y-NME.
- Force field: multi-ego-basic.ff at commit 5a3e97e8 (ideal bonded geometry), multi-eGO `--egos mg`, with every backbone φ/ψ dihedral removed (`strip_phipsi.py`).
- mg prior at commit 99b88459: `ATT_SIGMA="rrep"`, softened local O(i)–C(i+1) and C(i)–N(i+2) pairs, directional backbone H-bond (H–O 1.8 kJ/mol at 0.20 nm, N–O 0.9 at 0.293 nm, C–H 0.9 at 0.31 nm, i.e. scale ×0.90) and the Cα contact fix (`PAIR_RMIN`/`PAIR_EPS` in `type_definitions.py`). Changing these defaults changes the bare distributions, so the library has to be refitted.
- Box: 8 nm cubic.
- `bare_300K.mdp`: sd, dt 5 fs, tau_t 25 ps, 300 K, cut-off 2.5 nm, 200 ns per run, frames every 2.5 ps. The first 1 ns is dropped, leaving 79,600 frames per run.
- Convergence: half-run JSD has a median of 0.003 bits (max 0.005, CAP|TYR|GLY).
- Clipping: 553 of the 8,820 grids put more than 10% of the target or bare probability on cells clipped at ±6 kT, almost all with Gly or Pro at the centre (sparse NDRD statistics); the worst is ASP|GLY|LYS (0.50). Same set as for the ×0.65 prior (530 grids), so it comes from the target, not the prior.

**Previous parameterisation.** The ×0.65 prior (H–O 1.3, N–O/C–H 0.65) with its own library and bare data is tagged `prior-hb065` (commit 80d76320); this one is tagged `prior-hb090`. Both are kept so the two can be compared on the benchmarks before one is chosen.

**Check (28 Nov 2026 refit, ×0.90 prior).** Ac-(AAQAA)3-NH2, 303 K, 4 × 60 ns: DSSP H+G+I 0.19 (×0.65 with its own library: 0.10), mean αR/β/PPII/αL 58/15/23/4 (target 32/23/41/4), Rg 0.83 nm. On the Ac-(AAXAA)3-NH2 guest series (Lifson–Roig, vs Moreau et al. 2009) both variants give the same residue ranking: Gly and Pro right, little specificity among the other residues (r 0.25–0.34).

**Check (27 Nov 2026 refit, ×0.65 prior).** Ac-(AAQAA)3-NH2, mg prior, 303 K, 4 × 60 ns. Compared with the same prior on the 26 Nov library, the mean per-residue JSD to the TCB target drops from 0.086 to 0.043 bits. The mean αR/β/PPII/αL basins go from 58/11/27/4 to 51/19/27/4 (target 32/24/40/4), and DSSP H+G+I from 0.19 to 0.11 (experiment about 0.2 at 300 K). Rg goes from 0.83 to 0.86 nm. The old library did not cancel the single-residue αR preference that the directional H-bond pairs add, and that preference was propping up the helix.

**Check (26 Nov 2026 library, previous prior).** Reweighting the ideal-geometry AB42 MG prior at 278 K from the previous AB42-specific NDRD CMAP onto that library moved the mean basins αR/β/PPII/αL from 42/25/19/14 to 35/30/23/12 (NDRD target 31/30/26/13), and the mean JSD to the target from 0.031 to 0.019 bits.

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
- The NDRD library file (`NDRD_TCB.txt`, about 300 MB) is not included. Download it from https://dunbrack.fccc.edu/lab/ndrd.

Coil target: D. Ting, G. Wang, M. Shapovalov, R. Mitra, M. I. Jordan, R. L. Dunbrack Jr., *Neighbor-dependent Ramachandran probability distributions of amino acids developed from a hierarchical Dirichlet process model*, PLoS Comput Biol 6:e1000763 (2010).

## Licence and attribution

The Neighbor-Dependent Ramachandran Distributions (NDRD) are distributed by the Dunbrack lab under the Creative Commons Attribution 4.0 licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/legalcode; NDRD licence text: https://dunbrack.fccc.edu/lab/product_ndrd/ndrd_license.txt).

- **Adapted from NDRD:** `src/multiego/data/coil_cmap_300K.npz` and every `coil_cmap.itp` written by multi-eGO. The NDRD TCB densities were smoothed (σ 7.5°), floored, combined with the bare multi-eGO distributions into potentials at 300 K, clipped at ±6 kT and resampled on the 24×24 CMAP grid.
- **Attribution:** any public use or redistribution must acknowledge the NDRD, by a link to https://dunbrack.fccc.edu/lab/ndrd or by citing Ting et al., PLoS Comput Biol 6:e1000763 (2010), and must state that the NDRD is distributed under CC BY 4.0. The header of each `coil_cmap.itp` carries this notice; keep it.
- **Not derived from NDRD:** `data/bare_300K.npz` (bare multi-eGO distributions) and `src/multiego/data/backbone_terminal_cosines.json` (the force field's own cosines).
- **Older notice:** the 2010 NDRD files used for the fit still carry an older restrictive notice in their header, which predates the current CC BY 4.0 licence.

