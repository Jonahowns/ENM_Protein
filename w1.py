import models as m
import mod_protein as mod
import numpy as np



wdir = '/home/jonah/Dropbox (ASU)/Projects/COV/'

chain_coord, chain_exp_bfacts = m.get_pdb_info('RNAP_MOD.pdb', returntype=2, multimodel=True)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)
# print(fcoord)
# for i in range(10):
#     fcoord = np.delete(fcoord, -1)
#     ex_bfacts = np.delete(ex_bfacts, -1)

# fcoord = fcoord[:-15, :]
# ex_bfacts = ex_bfacts[:-15]

# print(fcoord.shape)
# del fcoord[-1]

anm = m.MVPANM(fcoord, ex_bfacts, T=300, cutoff=15, scale_resolution=25, k_factor=4)
anm.calc_mvp(cuda=True)
anm.mvp_theor_bfactors('rnap_cuda_mvp_mod_trial.png')

# hanm = m.HANM(fcoord, ex_bfacts, mcycles=1, ncycles=1, scale_factor=0.6)
# hanm.routine(cuda=False)
# hanm.hanm_theor_bfactors('rnap_hanm_trial.png')
# m.export_to_simulation(hanm, 'RNAP_MOD.pdb')


# mvp = m.MVPANM(fcoord, ex_bfacts, T=300, cutoff=20, scale_resolution=30, k_factor=5)
# mvp.calc_mvp(cuda=False)
# mvp.mvp_theor_bfactors(wdir + '1bu4_theor_mvp.png')

# def get_pdb_seq(pdbfile):
#     if "/" in pdbfile:
#         pdbid = pdbfile.rsplit('/', 1)[1].split('.')[0]
#     else:
#         pdbid = pdbfile.split('.')[0]
#     s1 = m.get_pdb_info(pdbid, pdbfile, returntype=3)
#     trash, seq = zip(*s1)
#     seq = ''.join(list(seq))
#     return seq
#
# def write_fasta(seqs, outfile, labels=[]):
#     o = open(outfile, 'w')
#     if len(labels) == len(seqs):
#         for i in range(len(seqs)):
#             print('>' + labels[i], file=o)
#             print(seqs[i], file=o)
#     else:
#         for i in range(len(seqs)):
#             print('>seq' + str(i+1), file=o)
#             print(seqs[i], file=o)
#
# vx = get_pdb_seq(wdir+'6vxx.pdb')
# vs = get_pdb_seq(wdir+'6vsb.pdb')
# vy = get_pdb_seq(wdir+'6vyb.pdb')
#
# seqs = [vx, vs, vy]
# write_fasta(seqs, wdir+'pdb.fasta', labels=['6vxx', '6vsb', '6vyb'])




# chains = mod.get_seq_by_chain('RNAP.pdb')
# mod.write_chain_seq(chains, pdbid='RNAP')

# mod.combine_fastas('pre_align_chainI.fasta', 'chainI.fasta', 'RNAPchainI.fasta')
# mod.conv_clustal_to_PIR('align_chaini.clustal', 'mod_chaini.pir')
# print(chainids)
# print(vx)
# mod.write_PIR(vx, wdir+'6vxx.pir', labels=['6vxx'])

# ndir = wdir + "COV_Modeller/"
# # mod.conv_clustal_to_PIR(ndir + 'clustal_align.txt', ndir+'COV_seqs.pir')
#



