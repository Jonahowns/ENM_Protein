import models as m

pep149 = m.peptide('149.pdb', cutoff=7, indx_cutoff=5, potential=7.0)
pep149.calc_bonds()
pep149.calc_a3s()
m.export_to_simulation(pep149, '149.pdb')