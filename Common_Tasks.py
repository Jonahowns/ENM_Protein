import models as m


def peptide_to_sim(pdbfile, cutoff=7, indx_cutoff=5, potential=7.0):
    pep = m.peptide(pdbfile, cutoff=cutoff, indx_cutoff=indx_cutoff, potential=potential)
    pep.calc_bonds()
    pep.calc_a3s()
    m.export_to_simulation(pep, pdbfile)

