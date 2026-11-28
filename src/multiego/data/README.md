# multi-eGO data files

- `coil_cmap_300K.npz`: the transferable coil CMAP library used by `multiego/cmap.py`.
  - 8,820 grids of 24×24 points in kJ/mol, keyed `L|X|R`.
  - Adapted from the Neighbor-Dependent Ramachandran Distributions (NDRD, https://dunbrack.fccc.edu/lab/ndrd; Ting et al., PLoS Comput Biol 6:e1000763, 2010), which are distributed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The NDRD TCB densities were smoothed and combined with bare multi-eGO distributions into potentials.
  - How it was made, the licence and the attribution requirements are in `tools/coil_cmap/README.md`.
- `backbone_terminal_cosines.json`: the backbone φ/ψ cosines of multi-ego-basic.ff before the coil CMAP. They are used for the chain-terminal torsions.
