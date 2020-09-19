import models as m

pep149 = m.peptide('227.pdb', cutoff=15, indx_cutoff=2, potential=10.0, backbone_weight=90.0)
# pep149.calc_bonds_bounded(11, 15)
pep149.calc_bonds()
pep149.calc_a3s()

m.export_to_simulation(pep149, '227.pdb')