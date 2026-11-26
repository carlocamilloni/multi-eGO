#!/usr/bin/env python3
"""Fit the transferable, neighbour-dependent coil CMAP library used by multi-eGO (src/multiego/data).

For a residue X with neighbours L and R (L or R = "CAP" for an ACE/NME-type cap):

    V(phi,psi | L,X,R) = -kT ln p_coil(L,X,R) + kT ln p_bare(L,X,R),     kT at T_fit (300 K)

p_coil : NDRD TCB library (Ting et al. PLoS Comput Biol 6:e1000763, 2010; https://dunbrack.fccc.edu/lab/ndrd;
         CC BY 4.0), neighbour-dependent:
         ln p(X|L,R) = ln p(X|L) + ln p(X|R) - ln p(X|ALL), renormalised (a CAP side uses no term).
p_bare : the bare multi-eGO MG force field (no backbone phi/psi terms) from capped-peptide runs, factorised
         the same way with the caps in the role of "ALL":
         ln p_bare(X|L,R) = ln p(ACE-L-X-NME: X) + ln p(ACE-X-R-NME: X) - ln p(ACE-X-NME: X).

Both are smoothed (sigma), floored and combined as V = -kT ln((p_coil+f)/(p_bare+f)), zero mean under p_coil,
clipped at +-vmax kT, and the 72x72 cells are averaged onto the 24x24 GROMACS grid corners (-180, -165, ...,
phi outer loop). This is the procedure of the AB42 prototype (AB42/dihedral_analysis/coil_cmap.py).

Subcommands
  bare    histogram the bare runs (<work>/runs/ace_*_nme/dih) -> bare npz (uint32 counts per component)
  fit     build the library for all 20 x 21 x 21 triplets (20 types + CAP on each side) -> library npz
  check   predict a run's phi/psi under the library by reweighting and compare with the NDRD target

See tools/coil_cmap/README.md for how the bare runs are produced.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coil_lib as cc  # noqa: E402

R_GAS, NB, CEN = cc.R_GAS, cc.NB, cc.CEN
ONE = dict(
    A="ALA",
    C="CYS",
    D="ASP",
    E="GLU",
    F="PHE",
    G="GLY",
    H="HIS",
    I="ILE",
    K="LYS",
    L="LEU",
    M="MET",
    N="ASN",
    P="PRO",
    Q="GLN",
    R="ARG",
    S="SER",
    T="THR",
    V="VAL",
    W="TRP",
    Y="TYR",
)
AA = sorted(ONE.values())
CAP = "CAP"


def key(left, x, right):
    return f"{left}|{x}|{right}"


def jsd(p, q):
    p = p.ravel() + 1e-12
    q = q.ravel() + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))


def basins(P):
    Pp, Ps = np.meshgrid(CEN, CEN, indexing="ij")
    aL = Pp > 0
    aR = (~aL) & (Ps > -120) & (Ps < 50)
    ext = (~aL) & ~aR
    return np.array([P[aR].sum(), P[ext & (Pp < -90)].sum(), P[ext & (Pp >= -90)].sum(), P[aL].sum()])


def downsample(V):
    """72 cell centres -> 24 CMAP grid corners at -180,-165,...: average of the 2x2 cells around each corner."""
    Vc = 0.25 * (V + np.roll(V, 1, 0) + np.roll(V, 1, 1) + np.roll(np.roll(V, 1, 0), 1, 1))
    return Vc[::3, ::3]


def bil(G, phi, psi):
    x, y = (phi + 180) / 15.0, (psi + 180) / 15.0
    i, j = np.floor(x).astype(int) % 24, np.floor(y).astype(int) % 24
    fx, fy = x - np.floor(x), y - np.floor(y)
    return (
        G[i, j] * (1 - fx) * (1 - fy)
        + G[(i + 1) % 24, j] * fx * (1 - fy)
        + G[i, (j + 1) % 24] * (1 - fx) * fy
        + G[(i + 1) % 24, (j + 1) % 24] * fx * fy
    )


# ----------------------------------------------------------------------------- bare
def cmd_bare(a):
    """(CAP,X,CAP) from ACE-X-NME; (L,X,CAP) and (CAP,X,R) from ACE-L-X-NME / ACE-X-R-NME."""
    H, half, nfr, missing = {}, {}, {}, []
    for d in sorted(glob.glob(os.path.join(a.work, "runs", "ace_*_nme"))):
        seq = os.path.basename(d).split("_")[1]
        dih = os.path.join(d, "dih")
        if not os.path.isdir(dih) or (not os.path.exists(os.path.join(d, "done")) and not a.partial):
            missing.append(seq)
            continue
        res = [ONE[c] for c in seq]
        for pos, x in enumerate(res):
            k = key(CAP, x, CAP) if len(res) == 1 else key(CAP, x, res[1]) if pos == 0 else key(res[0], x, CAP)
            f = glob.glob(os.path.join(dih, f"phi{x}{pos + 2}.xvg"))
            if not f:
                missing.append(f"{seq}:{x}{pos + 2}")
                continue
            phi, psi = cc.read_xvg(f[0], a.stride), cc.read_xvg(f[0].replace("/phi", "/psi"), a.stride)
            n = min(len(phi), len(psi))
            s0 = a.skip
            phi, psi = phi[s0:n], psi[s0:n]
            m = len(phi) // 2
            h1 = cc.gauss_smooth(cc.hist2d(phi[:m], psi[:m]), 5)
            h2 = cc.gauss_smooth(cc.hist2d(phi[m:], psi[m:]), 5)
            H[k], nfr[k], half[k] = cc.hist2d(phi, psi).astype(np.uint32), len(phi), jsd(h1, h2)
    np.savez_compressed(
        a.out,
        **H,
        _nframes=np.array([f"{k}={v}" for k, v in nfr.items()]),
        _halfjsd=np.array([f"{k}={v}" for k, v in half.items()]),
        _meta=np.array([f"work={os.path.abspath(a.work)} stride={a.stride} skip={a.skip}"]),
    )
    n0 = sum(k.startswith(CAP) and k.endswith(CAP) for k in H)
    print(f"wrote {a.out}: {len(H)} components ({n0} ACE-X-NME, {len(H) - n0} from pairs); missing: {len(missing)}")
    if half:
        v = np.array(list(half.values()))
        print(f"half-run JSD (bits): median {np.median(v):.4f}, max {v.max():.4f} ({max(half, key=half.get)})")


def load_bare(path):
    z = np.load(path)
    return {k: z[k].astype(float) for k in z.files if not k.startswith("_")}


# ----------------------------------------------------------------------------- fit
def smooth_norm(P, sigma):
    P = cc.gauss_smooth(P / P.sum(), sigma)
    P = np.maximum(P, 0)
    return P / P.sum()


def bare_triplet(B, left, x, right, sigma, floor):
    """p_bare(L,X,R) = p(L,X,cap) p(cap,X,R) / p(cap,X,cap); a CAP side (or a missing component) uses no term."""
    c0 = B.get(key(CAP, x, CAP))
    if c0 is None:
        return None, "no ACE-X-NME"
    p0 = smooth_norm(c0, sigma) + floor
    lp, how = np.log(p0), ""
    for k, tag, side in ((key(left, x, CAP), "L", left), (key(CAP, x, right), "R", right)):
        if side != CAP and k in B:
            lp += np.log(smooth_norm(B[k], sigma) + floor) - np.log(p0)
            how += tag
    p = np.exp(lp - lp.max())
    return p / p.sum(), how or "caps"


def cmd_fit(a):
    kT = R_GAS * a.temp
    lib = cc.read_ndrd(a.lib)
    B = load_bare(a.bare)
    floor = a.floor / (NB * NB)
    sides = AA + [CAP]
    G, info, clipmass = {}, {}, []
    for x in AA:
        if x in a.skip:
            continue
        for left in sides:
            for right in sides:
                k = key(left, x, right)
                pt, how_t = cc.coil_target(lib, x, left, right)
                if pt is None:
                    info[k] = "no NDRD"
                    continue
                pt = smooth_norm(pt, a.sigma)
                pb, how_b = bare_triplet(B, left, x, right, a.sigma, floor)
                if pb is None:
                    info[k] = how_b
                    continue
                V = -kT * np.log((pt + floor) / (pb + floor))
                V -= (pt * V).sum()  # gauge: zero mean under the target
                V = np.clip(V, -a.vmax * kT, a.vmax * kT)
                clip = np.abs(V) >= a.vmax * kT - 1e-9
                cm = max(pt[clip].sum(), pb[clip].sum())
                clipmass.append(cm)
                G[k] = downsample(V).astype(np.float32)
                info[k] = f"target={how_t or 'ALL'} bare={how_b} clipmass={cm:.2e}"
    meta = (
        f"T_fit={a.temp} sigma={a.sigma} floor={a.floor} vmax={a.vmax} units=kJ/mol grid=24x24 phi-outer from -180; "
        f"coil target: NDRD TCB (Ting et al. 2010); bare: {os.path.basename(a.bare)}"
    )
    np.savez_compressed(a.out, **G, _info=np.array([f"{k}={v}" for k, v in info.items()]), _meta=np.array([meta]))
    cm = np.array(clipmass)
    print(
        f"wrote {a.out}: {len(G)} grids; skipped {len(info) - len(G)}; probability on clipped cells: "
        f"median {np.median(cm):.1e}, max {cm.max():.1e}"
    )


# ----------------------------------------------------------------------------- check
def cmd_check(a):
    """p_new ∝ p_run exp(-(V_new - V_old)/kT) per residue at the run temperature; V_old = the run's CMAP grids
    (--old-grids, npz keyed by residue number) and/or the topology's phi/psi cosines."""
    kT = R_GAS * a.temp
    atoms, cos = cc.parse_top(a.top)
    seq, resids = cc.sequence(atoms)
    lib = cc.read_ndrd(a.lib)
    z = np.load(a.library)
    G = {k: z[k].astype(float) for k in z.files if not k.startswith("_")}
    old = {int(k): v for k, v in np.load(a.old_grids).items()} if a.old_grids else {}
    rows = []
    print(f"{'res':7s} {'JSD old':>7s} {'JSD new':>7s} {'ESS%':>5s} |  aR/beta/PPII/aL %: run -> predicted (target)")
    for k, r in enumerate(resids):
        if k == 0 or k == len(resids) - 1:
            continue
        t = key(
            cc.ALIAS.get(seq[k - 1], seq[k - 1]), cc.ALIAS.get(seq[k], seq[k]), cc.ALIAS.get(seq[k + 1], seq[k + 1])
        )
        f = glob.glob(os.path.join(a.dih, f"phi???{r}.xvg"))
        if t not in G or not f:
            continue
        phi, psi = cc.read_xvg(f[0], a.stride), cc.read_xvg(f[0].replace("/phi", "/psi"), a.stride)
        n = min(len(phi), len(psi))
        phi, psi = phi[:n], psi[:n]
        v_old = (
            (bil(old[r], phi, psi) if r in old else 0.0)
            + cc.vcos(cos.get(("phi", r)), phi)
            + cc.vcos(cos.get(("psi", r)), psi)
        )
        lw = -(bil(G[t], phi, psi) - v_old) / kT
        w = np.exp(lw - lw.max())
        w /= w.sum()
        Hold = cc.gauss_smooth(cc.hist2d(phi, psi), 5)
        Hold /= Hold.sum()
        Hnew = cc.gauss_smooth(cc.hist2d(phi, psi, w), 5)
        Hnew /= Hnew.sum()
        pt, _ = cc.coil_target(lib, seq[k], seq[k - 1], seq[k + 1])
        pts = cc.gauss_smooth(pt, 5)
        pts /= pts.sum()
        bo, bn, bt = basins(Hold) * 100, basins(Hnew) * 100, basins(pts) * 100
        rows.append((jsd(Hold, pts), jsd(Hnew, pts), bo, bn, bt, 1 / np.sum(w**2) / n))

        def fm(b):
            return "/".join(f"{x:.0f}" for x in b)

        print(
            f"{seq[k]}{r:<4d} {rows[-1][0]:7.3f} {rows[-1][1]:7.3f} {100 * rows[-1][5]:5.1f} |  "
            f"{fm(bo)} -> {fm(bn)} ({fm(bt)})"
        )
    if rows:
        J = np.array([[x[0], x[1]] for x in rows])
        Bs = np.array([[x[2], x[3], x[4]] for x in rows])
        print(
            f"mean JSD to target: run {J[:, 0].mean():.3f} -> predicted {J[:, 1].mean():.3f} bits; "
            f"min ESS {100 * min(x[5] for x in rows):.1f}%"
        )
        print(
            "mean basins aR/beta/PPII/aL: run",
            np.round(Bs[:, 0].mean(0)).astype(int),
            " predicted",
            np.round(Bs[:, 1].mean(0)).astype(int),
            " target",
            np.round(Bs[:, 2].mean(0)).astype(int),
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("bare")
    s.add_argument("--work", required=True, help="folder with runs/ace_*_nme (see setup_one.sh)")
    s.add_argument("--out", required=True)
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--skip", type=int, default=400, help="frames dropped at the start (400 = 1 ns at 2.5 ps)")
    s.add_argument("--partial", action="store_true", help="also use runs without the 'done' marker")
    s = sp.add_parser("fit")
    s.add_argument("--bare", required=True)
    s.add_argument(
        "--lib", required=True, help="NDRD_TCB.txt (https://dunbrack.fccc.edu/lab/ndrd, CC BY 4.0; not shipped: 300 MB)"
    )
    s.add_argument("--out", required=True)
    s.add_argument("--temp", type=float, default=300.0)
    s.add_argument("--sigma", type=float, default=7.5, help="smoothing, deg")
    s.add_argument("--floor", type=float, default=0.01, help="density floor, fraction of uniform")
    s.add_argument("--vmax", type=float, default=6.0, help="clip |V| (kT)")
    s.add_argument("--skip", nargs="*", default=[], help="residue types to leave out")
    s = sp.add_parser("check")
    s.add_argument("--library", required=True)
    s.add_argument("--lib", required=True, help="NDRD_TCB.txt")
    s.add_argument("--top", required=True)
    s.add_argument("--dih", required=True)
    s.add_argument("--temp", type=float, required=True)
    s.add_argument("--old-grids", help="npz of the run's current CMAP grids, keyed by residue number")
    s.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    {"bare": cmd_bare, "fit": cmd_fit, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
