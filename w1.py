import models as m
import numpy as np


wdir = '/home/jonah/Dropbox (ASU)/Projects/COV/'

chain_coord, chain_exp_bfacts = m.get_pdb_info('histone', wdir+'histone.pdb', returntype=2)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)
# anm = m.ANM(fcoord, ex_bfacts, T=300, cutoff=15)
# anm.calc_ANM_unitary(cuda=True)
# anm.anm_compare_bfactors(wdir+'1bu4newtrial.png')


mvp = m.MVPANM(fcoord, ex_bfacts, T=300, cutoff=20, scale_resolution=30, k_factor=5)
mvp.calc_mvp(cuda=False)
mvp.mvp_theor_bfactors(wdir + '1bu4_theor_mvp.png')

