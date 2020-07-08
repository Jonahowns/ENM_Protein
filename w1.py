import models as m
import numpy as np


wdir = '/home/jonah/Dropbox (ASU)/Projects/COV/'

chain_coord, chain_exp_bfacts = m.get_pdb_info('1bu4', wdir+'1bu4.pdb', returntype=2)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)
anm = m.ANM(fcoord, ex_bfacts, T=300, cutoff=15)
anm.calc_ANM_unitary(cuda=False)
anm.anm_compare_bfactors(wdir+'1bu4newtrial.png')