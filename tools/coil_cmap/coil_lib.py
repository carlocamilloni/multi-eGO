"""Shared utilities for the coil CMAP fit: 5-degree phi/psi grids, the NDRD library reader and the
neighbour-dependent coil target (Ting et al. PLoS Comput Biol 6:e1000763, 2010).
Copied from AB42/dihedral_analysis/coil_cmap.py (the AB42 prototype) so that tools/coil_cmap is self-contained."""

import os
import re

import numpy as np

R_GAS = 0.0083144626  # kJ/mol/K
NB = 72  # 5 deg cells
CEN = -177.5 + 5.0 * np.arange(NB)
ALIAS = {
    "HIE": "HIS",
    "HID": "HIS",
    "HIP": "HIS",
    "HSD": "HIS",
    "HSE": "HIS",
    "HSP": "HIS",
    "CYX": "CYS",
    "CYS2": "CYS",
}


def gauss_smooth(P, sigma_deg):
    """periodic 2D Gaussian smoothing of a 72x72 grid"""
    if sigma_deg <= 0:
        return P
    d = np.minimum(np.arange(NB), NB - np.arange(NB)) * 5.0
    k = np.exp(-0.5 * (d / sigma_deg) ** 2)
    k /= k.sum()
    K = np.fft.rfft2(np.outer(k, k))
    return np.fft.irfft2(np.fft.rfft2(P) * K, s=P.shape)


def hist2d(phi, psi, w=None):
    i = np.floor((np.asarray(phi) + 180.0) / 5.0).astype(int) % NB
    j = np.floor((np.asarray(psi) + 180.0) / 5.0).astype(int) % NB
    H = np.zeros((NB, NB))
    np.add.at(H, (i, j), 1.0 if w is None else w)
    return H


def read_xvg(path, stride=1):
    out = []
    with open(path) as f:
        for n, l in enumerate(x for x in f if x[0] not in "#@"):
            if n % stride == 0:
                out.append(float(l.split()[1]))
    return np.asarray(out)


def parse_top(path):
    """atoms {(resnr,name): (idx,type,resname)}, phi/psi cosine terms {('phi'|'psi', resnr): [(phase,k,n)]}"""
    atoms, sec, dih_lines = {}, None, []
    for line in open(path):
        s = line.split(";")[0].strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("["):
            sec = s.strip("[] ").strip()
            continue
        t = s.split()
        if sec == "atoms":
            atoms[(int(t[2]), t[4])] = (int(t[0]), t[1], t[3])
        elif sec == "dihedrals" and len(t) >= 8 and t[4] in ("1", "9"):
            dih_lines.append(t)
    by_idx = {v[0]: (k[0], k[1]) for k, v in atoms.items()}
    cos = {}
    for t in dih_lines:
        a = [by_idx[int(x)] for x in t[:4]]
        names = tuple(x[1] for x in a)
        if names == ("C", "N", "CA", "C"):
            key = ("phi", a[2][0])
        elif names == ("N", "CA", "C", "N"):
            key = ("psi", a[1][0])
        else:
            continue
        cos.setdefault(key, []).append((float(t[5]), float(t[6]), int(t[7])))
    return atoms, cos


def vcos(terms, x):
    return sum(k * (1 + np.cos(np.deg2rad(n * x - p))) for p, k, n in terms) if terms else np.zeros_like(x)


def sequence(atoms):
    res = {}
    for (r, n), (_, _, rn) in atoms.items():
        res[r] = ALIAS.get(rn, rn)
    return [res[r] for r in sorted(res)], sorted(res)


def read_ndrd(path):
    """returns lib[(res, dir, neigh)] = 72x72 probability grid on CEN cells; dir in {'left','right','ALL'}.

    Real NDRD files give 5x5 bin CENTRES at -180..175 (readme), i.e. bins straddle our cells -> each value is
    split over the 2x2 cells it overlaps (a 2.5 deg shift + 5 deg box, negligible vs the fit smoothing).
    The stand-in library (header 'STAND-IN') was written with bin FLOORS -> mapped directly.
    Parsed grids are cached in <path>.cache.npz (licensed data: keep it with the original files)."""
    cache = path + ".cache.npz"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        z = np.load(cache)
        return {tuple(k.split("|")): z[k] for k in z.files}
    with open(path) as f:
        head = "".join(next(f, "") for _ in range(5))
    centres = "STAND-IN" not in head
    lib = {}
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) < 6 or t[0].startswith("#") or not re.match(r"^-?\d", t[3]):
                continue
            res, d, nb = t[0].upper(), t[1], t[2].upper()
            key = (res, d if d == "ALL" else d.lower(), nb)
            if key not in lib:
                lib[key] = np.zeros((NB, NB))
            lib[key][int(round((float(t[3]) + 180) / 5)) % NB, int(round((float(t[4]) + 180) / 5)) % NB] += float(t[5])
    for k in lib:
        P = lib[k]
        if centres:  # value at centre c covers [c-2.5, c+2.5] = half of cell (c-5..c) and half of cell (c..c+5)
            P = 0.25 * (P + np.roll(P, -1, 0) + np.roll(P, -1, 1) + np.roll(np.roll(P, -1, 0), -1, 1))
        lib[k] = P / P.sum()
    np.savez(cache, **{"|".join(k): v for k, v in lib.items()})
    return lib


def ndrd_all(lib, res):
    """neighbour-independent map: stand-in 'ALL ALL', real NDRD 'right ALL' (the term subtracted in the
    NDRD triplet formula; 'left ALL' differs slightly because of chain ends)"""
    for key in [(res, "ALL", "ALL"), (res, "right", "ALL"), (res, "left", "ALL")]:
        if key in lib:
            return lib[key]
    return None


def coil_target(lib, res, left, right, neighbours=True):
    """p(phi,psi | res, left, right) following Ting et al. 2010, with fallback to the ALL map"""
    pall = ndrd_all(lib, res)
    if pall is None:
        return None, "missing"
    if not neighbours:
        return pall, "ALL"
    pl, pr = lib.get((res, "left", left)), lib.get((res, "right", right))
    if pl is None and pr is None:
        return pall, "ALL"
    eps = 1e-12
    lp = np.log(pall + eps)
    if pl is not None:
        lp += np.log(pl + eps) - np.log(pall + eps)
    if pr is not None:
        lp += np.log(pr + eps) - np.log(pall + eps)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return p, ("L" if pl is not None else "") + ("R" if pr is not None else "")
