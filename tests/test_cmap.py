"""
Unit tests for multiego.cmap (coil CMAP backbone term).

A minimal ensemble-like object is built by hand: residues with backbone atoms N, CA, C (+ O), consecutive
residues bonded C(i-1)-N(i), and backbone phi/psi dihedrals as pdb2gmx would write them.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multiego import cmap

BB = ["N", "CA", "C", "O"]


def _ensemble(resnames, phi_psi_k=0.0, caps=()):
    """Linear chain; caps: residues (by index) that are ACE (C, O only) or NME (N, CA only)."""
    rows, atom = [], {}
    n = 0
    for r, rn in enumerate(resnames, 1):
        names = ["CA", "C", "O"] if rn == "ACE" else ["N", "CA"] if rn in ("NME", "NH2") else BB
        for name in names:
            n += 1
            atom[(r, name)] = n
            rows.append(
                {
                    "number": str(n),
                    "resnum": str(r),
                    "resname": rn,
                    "name": name,
                    "sb_type": f"{name}_P_{r}",
                    "molecule_name": "P",
                }
            )
    bonds = []
    for r in range(1, len(resnames) + 1):
        if (r, "N") in atom and (r, "CA") in atom:
            bonds.append((atom[(r, "N")], atom[(r, "CA")]))
        if (r, "CA") in atom and (r, "C") in atom:
            bonds.append((atom[(r, "CA")], atom[(r, "C")]))
        if (r, "C") in atom and (r + 1, "N") in atom:
            bonds.append((atom[(r, "C")], atom[(r + 1, "N")]))
    dih = []
    for r in range(1, len(resnames) + 1):
        if resnames[r - 1] in ("ACE", "NME", "NH2"):
            continue
        if (r - 1, "C") in atom:
            dih.append((atom[(r - 1, "C")], atom[(r, "N")], atom[(r, "CA")], atom[(r, "C")]))
        if (r + 1, "N") in atom:
            dih.append((atom[(r, "N")], atom[(r, "CA")], atom[(r, "C")], atom[(r + 1, "N")]))
    dihedrals = pd.DataFrame(
        [
            {"ai": a, "aj": b, "ak": c, "al": d, "funct": 1, "phase": 0.0, "phi_k": phi_psi_k, "per": 1}
            for a, b, c, d in dih
        ]
    )
    bonded = {"P": {"bonds": pd.DataFrame(bonds, columns=["ai", "aj"]).assign(funct=1), "dihedrals": dihedrals}}
    ens = SimpleNamespace(
        topology_dataframe=pd.DataFrame(rows), molecule_type_dict={"P": "protein"}, meGO_bonded_interactions=bonded
    )
    return ens, atom


@pytest.fixture(scope="module")
def library():
    return cmap.load_library()[0]


@pytest.fixture(scope="module")
def cosines():
    with open(cmap.TERMINAL_COSINES) as f:
        return json.load(f)


def test_library_complete(library):
    aa = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
    sides = aa + ["CAP"]
    assert len(library) == 20 * 21 * 21
    assert all(f"{lft}|{x}|{rgt}" in library for x in aa for lft in sides for rgt in sides)
    assert all(g.shape == (24, 24) and np.isfinite(g).all() for g in library.values())


def test_middle_residue_gets_cmap_and_placeholders_go(library, cosines):
    ens, atom = _ensemble(["ALA", "GLY", "SER"])
    out = cmap.build_backbone_terms(ens)["molecules"]["P"]
    assert len(out["types"]) == 1
    five, grid, label = out["types"][0]
    assert label.endswith("ALA|GLY|SER")
    np.testing.assert_allclose(grid, library["ALA|GLY|SER"])
    assert list(out["cmap"].iloc[0][["ai", "aj", "ak", "al", "am"]]) == [
        atom[(1, "C")],
        atom[(2, "N")],
        atom[(2, "CA")],
        atom[(2, "C")],
        atom[(3, "N")],
    ]
    assert five == ["C_P_1", "N_P_2", "CA_P_2", "C_P_2", "N_P_3"]
    dih = ens.meGO_bonded_interactions["P"]["dihedrals"]
    # the placeholders are gone; terminal psi(ALA1) and phi(SER3) carry exactly the original rtp cosines
    psi1 = (atom[(1, "N")], atom[(1, "CA")], atom[(1, "C")], atom[(2, "N")])
    phi3 = (atom[(2, "C")], atom[(3, "N")], atom[(3, "CA")], atom[(3, "C")])
    for quad, terms in ((psi1, cosines["ALA"]["psi"]), (phi3, cosines["SER"]["phi"])):
        rows = dih[(dih[["ai", "aj", "ak", "al"]].to_numpy() == np.array(quad)).all(axis=1)]
        got = sorted(zip(rows["phase"], rows["phi_k"], rows["per"]))
        assert got == sorted((t["phase"], t["k"], t["mult"]) for t in terms)
    assert len(dih) == out["added"] == len(cosines["ALA"]["psi"]) + len(cosines["SER"]["phi"])
    assert out["removed"] == 4  # 2 placeholders of GLY2 + the 2 terminal placeholders
    assert out["missing"] == []


def test_real_input_cosines_replaced_for_cmap_and_kept_at_termini():
    ens, atom = _ensemble(["ALA", "GLY", "SER"], phi_psi_k=2.0)
    out = cmap.build_backbone_terms(ens)["molecules"]["P"]
    dih = ens.meGO_bonded_interactions["P"]["dihedrals"]
    assert out["removed"] == 2 and out["added"] == 0
    kept = {tuple(x) for x in dih[["ai", "aj", "ak", "al"]].to_numpy()}
    assert (atom[(1, "N")], atom[(1, "CA")], atom[(1, "C")], atom[(2, "N")]) in kept
    assert (atom[(2, "N")], atom[(2, "CA")], atom[(2, "C")], atom[(3, "N")]) not in kept


def test_caps_are_cap_neighbours(library):
    ens, _ = _ensemble(["ACE", "ALA", "NME"])
    out = cmap.build_backbone_terms(ens)["molecules"]["P"]
    assert [t[2].split()[1] for t in out["types"]] == ["CAP|ALA|CAP"]
    assert out["added"] == 0
    assert ens.meGO_bonded_interactions["P"]["dihedrals"].empty


def test_aliases_and_unknown_residues(library):
    ens, _ = _ensemble(["HISE", "CYS2", "XYZ", "ALA"])
    out = cmap.build_backbone_terms(ens)["molecules"]["P"]
    labels = [t[2].split()[1] for t in out["types"]]
    assert labels == ["HIS|CYS|CAP"]  # XYZ has no grid: right neighbour becomes CAP, XYZ itself gets none
    assert any(m.startswith("XYZ3") for m in out["missing"])


def test_non_protein_untouched():
    ens, _ = _ensemble(["ALA", "GLY", "SER"])
    ens.molecule_type_dict = {"P": "other"}
    before = ens.meGO_bonded_interactions["P"]["dihedrals"].copy()
    out = cmap.build_backbone_terms(ens)
    assert out["molecules"] == {}
    pd.testing.assert_frame_equal(ens.meGO_bonded_interactions["P"]["dihedrals"], before)


def test_write_cmap_itp(tmp_path, library):
    ens, _ = _ensemble(["ALA", "GLY", "SER"])
    terms = cmap.build_backbone_terms(ens)
    path = tmp_path / "coil_cmap.itp"
    cmap.write_cmap_itp(terms, str(path), "; header")
    text = path.read_text()
    assert "[ cmaptypes ]" in text
    body = text.split("C_P_1 N_P_2 CA_P_2 C_P_2 N_P_3 1 24 24\\\n")[1].split("\n\n")[0]
    values = np.array([float(x) for x in body.replace("\\", " ").split()])
    np.testing.assert_allclose(values, library["ALA|GLY|SER"].ravel(), atol=1e-6)


def test_terminal_cosines_table_matches_ffbonded(cosines):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    defs = {}
    for line in open(os.path.join(root, "multi-ego-basic.ff", "ffbonded.itp")):
        t = line.split(";")[0].split()
        if len(t) >= 5 and t[0] == "#define":
            defs[t[1]] = (float(t[2]), float(t[3]), int(t[4]))
    for res, d in cosines.items():
        if res.startswith("_"):
            continue
        for kind in ("phi", "psi"):
            for term in d[kind]:
                assert defs[term["define"]] == (term["phase"], term["k"], term["mult"])
