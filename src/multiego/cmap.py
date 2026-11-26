"""
Coil CMAP backbone term.

multi-ego-basic.ff no longer defines backbone phi/psi proper dihedrals: each protein residue X that has both
backbone neighbours gets a neighbour-dependent CMAP V(phi, psi | L, X, R) from the transferable library in
``data/coil_cmap_300K.npz``. The library was fitted at 300 K against the NDRD TCB coil library (Ting et al.
PLoS Comput Biol 6:e1000763, 2010), using bare multi-eGO MG references of ACE-X-NME and ACE-X-Y-NME peptides
(see tools/coil_cmap). L or R is ``CAP`` when the neighbour is a cap (ACE, NME, NH2) or a residue without
library data.

In multi-ego-basic.ff each residue carries one zero-strength placeholder per backbone torsion (``gd_cmap``), which
only stops pdb2gmx from generating unparametrised torsions. For every residue that gets a CMAP, all backbone phi/psi
proper dihedrals of the input topology (placeholders, or real cosines from a topology made with the old force
field) are removed. Torsions that a CMAP cannot cover (psi of a chain's first residue, phi of its last one, and
residues without a library grid) get the original per-residue rtp cosines from
``data/backbone_terminal_cosines.json``, unless the input topology already gives them a non-zero term.
"""

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LIBRARY = os.path.join(DATA_DIR, "coil_cmap_300K.npz")
TERMINAL_COSINES = os.path.join(DATA_DIR, "backbone_terminal_cosines.json")
CAP = "CAP"
CAP_RESIDUES = {"ACE", "NME", "NH2", "NAC", "CT3"}
ALIAS = {
    "HIE": "HIS",
    "HID": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "HISD": "HIS",
    "HISE": "HIS",
    "HISH": "HIS",
    "CYS2": "CYS",
    "CYX": "CYS",
    "LYSN": "LYS",
}
CMAP_COLUMNS = ["ai", "aj", "ak", "al", "am", "funct"]


@lru_cache(maxsize=4)
def load_library(path=LIBRARY):
    """Return ({'L|X|R': 24x24 grid in kJ/mol}, metadata string)."""
    z = np.load(path)
    grids = {k: z[k].astype(float) for k in z.files if not k.startswith("_")}
    meta = str(z["_meta"][0]) if "_meta" in z.files else ""
    return grids, meta


@lru_cache(maxsize=4)
def load_terminal_cosines(path=TERMINAL_COSINES):
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _canonical(resname):
    return ALIAS.get(resname, resname)


def _torsion_mask(dihedrals, quad):
    """Rows of a dihedrals DataFrame that are proper (funct 1/9) torsions over the atom quadruple (either direction)."""
    if dihedrals.empty:
        return pd.Series(False, index=dihedrals.index)
    idx = dihedrals[["ai", "aj", "ak", "al"]].astype(int).to_numpy()
    q = np.array(quad)
    hit = np.all(idx == q, axis=1) | np.all(idx == q[::-1], axis=1)
    return pd.Series(hit, index=dihedrals.index) & dihedrals["funct"].astype(int).isin([1, 9])


def build_backbone_terms(meGO_ensemble, library_path=LIBRARY):
    """
    Assign the coil CMAP to every protein residue with both backbone neighbours and fix the remaining backbone
    torsions. Modifies ``meGO_ensemble.meGO_bonded_interactions[molecule]['dihedrals']`` in place.

    Returns
    -------
    dict
        ``{"meta": str, "molecules": {molecule: {"cmap": DataFrame(ai, aj, ak, al, am, funct),
        "types": [(5 sb_types, 24x24 grid, label)], "removed": int, "added": int, "missing": [str]}}}``
    """
    grids, meta = load_library(library_path)
    cosines = load_terminal_cosines()
    topo = meGO_ensemble.topology_dataframe
    out = {"meta": meta, "molecules": {}}
    for molecule, moltype in meGO_ensemble.molecule_type_dict.items():
        if moltype != "protein" or molecule not in meGO_ensemble.meGO_bonded_interactions:
            continue
        bonded = meGO_ensemble.meGO_bonded_interactions[molecule]
        mol = topo.loc[topo["molecule_name"] == molecule]
        atom = {
            (int(r), n): (int(i), t) for r, n, i, t in mol[["resnum", "name", "number", "sb_type"]].itertuples(False)
        }
        resname = {int(r): rn for r, rn in mol[["resnum", "resname"]].drop_duplicates().itertuples(False)}
        bonds = set()
        for ai, aj in bonded["bonds"][["ai", "aj"]].astype(int).itertuples(False):
            bonds.add((ai, aj))
            bonds.add((aj, ai))

        def linked(r_prev, r_next):
            c, n = atom.get((r_prev, "C")), atom.get((r_next, "N"))
            return c is not None and n is not None and (c[0], n[0]) in bonds

        dihedrals = bonded["dihedrals"]
        drop = pd.Series(False, index=dihedrals.index)
        cmap_rows, types, added, missing = [], [], [], []
        for r in sorted(resname):
            x = _canonical(resname[r])
            if x in CAP_RESIDUES or (r, "CA") not in atom or (r, "N") not in atom or (r, "C") not in atom:
                continue
            has_left = linked(r - 1, r)
            has_right = linked(r, r + 1)
            left = _canonical(resname[r - 1]) if has_left else None
            right = _canonical(resname[r + 1]) if has_right else None
            phi = [atom[(r - 1, "C")], atom[(r, "N")], atom[(r, "CA")], atom[(r, "C")]] if has_left else None
            psi = [atom[(r, "N")], atom[(r, "CA")], atom[(r, "C")], atom[(r + 1, "N")]] if has_right else None
            label = None
            if has_left and has_right:
                lkey = left if f"{left}|{x}|{CAP}" in grids else CAP
                rkey = right if f"{CAP}|{x}|{right}" in grids else CAP
                label = f"{lkey}|{x}|{rkey}"
                if label not in grids:
                    label = None
            if label is not None:
                five = phi + [psi[-1]]
                cmap_rows.append([a[0] for a in five] + [1])
                types.append(([a[1] for a in five], grids[label], f"{resname[r]}{r} {label}"))
                for quad in (phi, psi):
                    drop |= _torsion_mask(dihedrals, [a[0] for a in quad])
                continue
            # no CMAP: keep existing phi/psi terms, add the original rtp cosines where none (or only the
            # zero-strength gd_cmap placeholders of multi-ego-basic.ff) are defined
            for kind, quad in (("phi", phi), ("psi", psi)):
                if quad is None:
                    continue
                present = _torsion_mask(dihedrals, [a[0] for a in quad])
                if present.any() and (dihedrals.loc[present, "phi_k"].astype(float).abs() > 0).any():
                    continue
                drop |= present
                terms = cosines.get(resname[r], cosines.get(x, {})).get(kind, [])
                if not terms:
                    missing.append(f"{resname[r]}{r} {kind}")
                for t in terms:
                    added.append(
                        {
                            "ai": quad[0][0],
                            "aj": quad[1][0],
                            "ak": quad[2][0],
                            "al": quad[3][0],
                            "funct": 1,
                            "phase": t["phase"],
                            "phi_k": t["k"],
                            "per": t["mult"],
                        }
                    )
        removed = int(drop.sum())
        dihedrals = dihedrals.loc[~drop]
        if added:
            dihedrals = pd.concat([dihedrals, pd.DataFrame(added)], ignore_index=True)
        bonded["dihedrals"] = dihedrals.reset_index(drop=True)
        out["molecules"][molecule] = {
            "cmap": pd.DataFrame(cmap_rows, columns=CMAP_COLUMNS),
            "types": types,
            "removed": removed,
            "added": len(added),
            "missing": missing,
        }
    return out


def write_cmap_itp(backbone_terms, path, header=""):
    """Write the [ cmaptypes ] of all molecules (24x24 grids, phi outer loop, from -180 deg)."""
    lines = [
        header.rstrip(),
        "; coil CMAP (multiego/cmap.py): " + backbone_terms["meta"],
        "; coil target: NDRD TCB library, Ting et al. PLoS Comput Biol 6:e1000763 (2010)",
        "",
        "[ cmaptypes ]",
        "",
    ]
    for molecule, data in backbone_terms["molecules"].items():
        for five, grid, label in data["types"]:
            v = np.asarray(grid, float).ravel()
            rows = [v[i:j] for i, j in zip(range(0, len(v), 10), range(10, len(v) + 10, 10))]
            body = "\\\n".join(" ".join(f"{x:.6f}" for x in row) for row in rows)
            lines.append(f"; {molecule} {label}\n" + " ".join(five) + " 1 24 24\\\n" + body + "\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))
