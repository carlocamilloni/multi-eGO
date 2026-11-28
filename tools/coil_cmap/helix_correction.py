"""Residue-specific helix-propensity correction on top of the coil CMAP library.

For each central residue X, every grid L|X|R gets V' = V + c_X * m(phi, psi), with m a smooth version of the
helical box of Best, de Sancho & Mittal 2012 (-100<phi<-30, -67<psi<-7; logistic edges, width --edge deg) evaluated on the CMAP nodes.
c_X is solved so that the single-residue helix free energy in A|X|A, G_X = -RT ln p_hel (model = bare x CMAP),
reproduces Moreau et al. (2009) ddG relative to Ala at --temp (interpolated between their 25 and 60 C data):
    G'_X - G'_A = ddG_Moreau(X).
The absolute level is fixed by --anchor: 'aaqaa' keeps 12 G_A + 3 G_Q unchanged (AAQAA helicity, to which the
H-bond scale was tuned, stays put); 'ala' keeps Ala unchanged.  (On Ac-(AAXAA)3-NH2 the simulated guest ddG follows
this single-residue estimate (r 0.95 without the correction), and with it matches Moreau with r 0.97; see README.md.)
Pro is left uncorrected by default (--skip PRO): its helix breaking is nonlocal (no amide H, constrains i-1), and the
single-residue estimate does not describe it (local +0.26 vs simulated +1.36 kcal/mol).
usage: python helix_correction.py --library <uncorrected fit>.npz --bare data/bare_300K.npz \
           --out ../../src/multiego/data/coil_cmap_300K.npz [--temp 300] [--anchor aaqaa] [--skip PRO]
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coil_lib as cc  # noqa: E402
from fit_coil_cmap import bil, bare_triplet, load_bare, key  # noqa: E402

MOREAU = os.path.join(HERE, "data", "moreau2009_helix.dat")
RES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def moreau(T):
    Tc = T - 273.15
    out = {}
    for line in open(MOREAU):
        if line.strip() and line[0] != "#":
            f = line.split()
            g = dict(zip((2.0, 25.0, 60.0), (float(f[4]), float(f[6]), float(f[8]))))
            lo, hi = (2.0, 25.0) if Tc <= 25 else (25.0, 60.0)
            out[f[0]] = g[lo] + (g[hi] - g[lo]) * (min(max(Tc, lo), hi) - lo) / (hi - lo)
    return out


def mask(phi, psi, w):
    s = lambda x: 1.0 / (1.0 + np.exp(-x / w))
    return s(phi + 100) * s(-30 - phi) * s(psi + 67) * s(-7 - psi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--bare", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--anchor", choices=["aaqaa", "ala"], default="aaqaa")
    ap.add_argument("--edge", type=float, default=7.5)
    ap.add_argument(
        "--skip", default="PRO", help="comma-separated residues left uncorrected (Pro: helix breaking is nonlocal)"
    )
    a = ap.parse_args()
    RT = 1.987204e-3 * a.temp
    kT = 8.314462618e-3 * a.temp
    Z = np.load(a.library)
    B = load_bare(a.bare)
    node = -180.0 + 15.0 * np.arange(24)
    MG = mask(*np.meshgrid(node, node, indexing="ij"), a.edge)
    P, S = np.meshgrid(cc.CEN, cc.CEN, indexing="ij")
    hel = (P > -100) & (P < -30) & (S > -67) & (S < -7)
    MB = bil(MG, P, S)

    pb, V = {}, {}
    for X in RES:
        pb[X], _ = bare_triplet(B, "ALA", X, "ALA", 7.5, 0.01 / (cc.NB * cc.NB))
        V[X] = bil(Z[key("ALA", X, "ALA")].astype(float), P, S)

    def G(X, c):
        q = pb[X] * np.exp(-(V[X] + c * MB - V[X].min()) / kT)
        return -RT * np.log(q[hel].sum() / q.sum())

    G0 = {X: G(X, 0.0) for X in RES}
    mo = moreau(a.temp)
    D = {X: (mo[X] - (G0[X] - G0["ALA"])) for X in RES}  # needed change of G_X - G_A
    dA = -3.0 * D["GLN"] / 15.0 if a.anchor == "aaqaa" else 0.0
    delta = {X: D[X] + dA for X in RES}
    delta["ALA"] = dA
    skip = set(filter(None, a.skip.split(",")))
    c = {X: 0.0 if X in skip else brentq(lambda cc_: G(X, cc_) - G0[X] - delta[X], -40, 40) for X in RES}

    out = {}
    for k in Z.files:
        if k.startswith("_"):
            continue
        x = k.split("|")[1]
        out[k] = (Z[k].astype(float) + c[x] * MG).astype(Z[k].dtype)
    meta = (
        (str(Z["_meta"][0]) if "_meta" in Z.files else "")
        + f" | helix correction (helix_correction.py): Moreau 2009 ddG at {a.temp:.0f} K, anchor {a.anchor}, edge {a.edge}, skip {a.skip}"
    )
    out["_meta"] = np.array([meta])
    if "_info" in Z.files:
        out["_info"] = Z["_info"]
    out["_helixcorr"] = np.array([f"{X} c={c[X]:+.3f}kJ/mol delta={delta[X]:+.3f}kcal/mol" for X in RES])
    np.savez_compressed(a.out, **out)

    print(f"# T {a.temp} K, anchor {a.anchor}; c_X added on the helical core (kJ/mol at m=1)")
    print("# X     p_hel before  G-G_A before  Moreau   c_X(kJ/mol)  G-G_A after")
    G1 = {X: G(X, c[X]) for X in RES}
    for X in RES:
        print(
            f"{X}   {np.exp(-G0[X]/RT):.3f}       {G0[X]-G0['ALA']:+.2f}        {mo[X]:.2f}    {c[X]:+6.2f}       {G1[X]-G1['ALA']:+.2f}"
        )
    print(f"# AAQAA anchor check 12 dG_A + 3 dG_Q = {12*(G1['ALA']-G0['ALA'])+3*(G1['GLN']-G0['GLN']):+.3f} kcal/mol")
    g = np.array([out[k] for k in out if not k.startswith("_")])
    print(f"# grid range after {g.min():.1f} .. {g.max():.1f} kJ/mol (library clip was +-{6*8.314462618e-3*300:.1f})")


if __name__ == "__main__":
    main()
