"""Build capped peptides ACE-X-NME / ACE-X-Y-NME (heavy atoms; pdb2gmx adds the polar H).
Built as G-X(-Y)-G with PeptideBuilder in an extended conformation, then the first Gly is cut to ACE (CA, C, O)
and the last to NME (N, CA).  usage: python build.py <seq one-letter, e.g. A or AW> <out.pdb>"""

import sys
from PeptideBuilder import Geometry
import PeptideBuilder
from Bio.PDB import PDBIO

seq, out = sys.argv[1], sys.argv[2]
full = "G" + seq + "G"
geos = []
for aa in full:
    g = Geometry.geometry(aa)
    g.phi, g.psi_im1 = -120.0, 130.0
    geos.append(g)
s = PeptideBuilder.initialize_res(geos[0])
for g in geos[1:]:
    PeptideBuilder.add_residue(s, g)
chain = list(s[0])[0]
res = list(chain)
keep = {0: ("ACE", {"CA", "C", "O"}), len(res) - 1: ("NME", {"N", "CA"})}
for k, r in enumerate(res):
    if k in keep:
        name, atoms = keep[k]
        for a in [a.get_id() for a in r if a.get_id() not in atoms]:
            r.detach_child(a)
        r.resname = name
    for a in [a.get_id() for a in r if a.get_id() == "OXT"]:
        r.detach_child(a)
io = PDBIO()
io.set_structure(s)
io.save(out)
