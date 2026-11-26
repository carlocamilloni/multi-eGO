"""Remove the backbone phi/psi proper dihedrals (funct 1/9) of every non-cap residue from a multi-eGO MG topology.
phi = C(i-1) N CA C, psi = N CA C N(i+1). Everything else (omega, impropers, side chains, pairs, LJ) is kept.
usage: python strip_phipsi.py <topol_mego.top> <out.top>"""

import sys

top, out = sys.argv[1], sys.argv[2]
atoms, sec, kept, removed = {}, None, [], {}
for line in open(top):
    s = line.split(";")[0].strip()
    if s.startswith("["):
        sec = s.strip("[] ").strip()
    elif sec == "atoms" and s and not s.startswith("#"):
        t = s.split()
        atoms[int(t[0])] = (int(t[2]), t[3], t[4])
    elif sec == "dihedrals" and s and not s.startswith("#"):
        t = s.split()
        if len(t) >= 5 and t[4] in ("1", "9"):
            a = [atoms[int(x)] for x in t[:4]]
            names = tuple(x[2] for x in a)
            r = a[2] if names == ("C", "N", "CA", "C") else a[1] if names == ("N", "CA", "C", "N") else None
            if r is not None and r[1] not in ("ACE", "NME"):
                key = (r[0], r[1], "phi" if names[0] == "C" else "psi")
                removed[key] = removed.get(key, 0) + 1
                continue
    kept.append(line)
open(out, "w").writelines(kept)
print(out, "removed:", ", ".join(f"{k[1]}{k[0]} {k[2]} x{v}" for k, v in sorted(removed.items())))
