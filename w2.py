import models as m
import mod_protein as mod
import numpy as np

chain_coord, chain_exp_bfacts, a1s = m.get_pdb_info('histone.pdb', returntype=4, multimodel=True, orientations=True)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)
at = m.ANMT(fcoord, a1s, ex_bfacts)
at.calc_a3s()
at.ana_gamma = 5.0
at.cutoff = 12.0

m.export_to_simulation(at, 'histone.pdb')


# anm = m.ANM(fcoord, ex_bfacts, T=100, cutoff=30)
# anm.calc_ANM_unitary(cuda=False)
# anm.anm_compare_bfactors('histone_trial.png')