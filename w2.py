import models as m
import mod_protein as mod
import numpy as np

chain_coord, chain_exp_bfacts = m.get_pdb_info('histone.pdb', returntype=2, multimodel=True)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)


anm = m.ANM(fcoord, ex_bfacts, T=100, cutoff=30)
anm.calc_ANM_unitary(cuda=False)
anm.anm_compare_bfactors('histone_trial.png')