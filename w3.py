import models as m
import numpy as np
import matplotlib.pyplot as plt

chain_coord, chain_exp_bfacts, normal_vectors = m.get_pdb_info('1bu4.pdb', returntype=4, multimodel=False, orientations=True)
fcoord = np.asarray(m.flatten(chain_coord))
ex_bfacts = m.flatten(chain_exp_bfacts)
# #
sim5 = 5*0.05709
sim7 = 7*0.05709
sim15 = 15*0.05709
ANMT = m.ANMT(fcoord, normal_vectors, ex_bfacts, T=300)
ANMT.calc_a3s()
bh = ANMT.calc_bend_hess(kb=sim15, kt=0)
torh = ANMT.calc_bend_hess(kb=0, kt=sim15)
bothh = ANMT.calc_bend_hess(kb=sim15, kt=sim15)



anm = m.ANM(fcoord, ex_bfacts, cutoff=10, T=300)
anm.calc_dist_matrix()
ah = anm.calc_hess_fast_unitary()
icheck = anm.calc_inv_Hess(ah, cuda=False)
#
# th = ANMT.total_hess(bh, ah, ks=sim5)
# h2 = ANMT.total_hess(torh, ah, ks=sim5)
# btoth = ANMT.total_hess(bothh, ah, ks=sim5)
# ih = ANMT.calc_inv_Hess(th, cuda=False)
# ih2 = ANMT.calc_inv_Hess(h2, cuda=False)
# bih = ANMT.calc_inv_Hess(btoth, cuda=False)
# bf = ANMT.calc_bfacts(ih)
# tf = ANMT.calc_bfacts(ih2)
# bothf = ANMT.calc_bfacts(bih)
# print(bf)
# print(tf)
#
# p = m.protein('1bu4.pdb', cutoff=7, pottype='s', potential=7.0, Angle_Constraint=True, a1s=ANMT.normal_vectors, a3s=ANMT.a3s)
# p.WriteSimFiles()

# s0msd, Normalbfact = m.load_sim_rmsds_from_file('5ks.json')
# s1msd, s1bfacts = m.load_sim_rmsds_from_file('justTor.json')
# s3msd, s3bfacts = m.load_sim_rmsds_from_file('both5t2.json')
# s2msd, s2bfacts = m.load_sim_rmsds_from_file('bend15.json')
# s4msd, s4bfacts = m.load_sim_rmsds_from_file('both15.json')
# s5msd, s5bfacts = m.load_sim_rmsds_from_file('15tor.json')
# m.free_compare('Both5_t4.png', s3bfacts, s5bfacts, s2bfacts, s4bfacts, ex_bfacts, Normalbfact, legends=['5kb5kt', '15kt', '15kb', '15kbkt', 'Exp', 'Normal k=5.0'], title='ACT Bfactor Comparison')

# s1msd, s1bfacts = m.load_sim_rmsds_from_file('5kb0kt.json')
# m.free_compare('5kb0kt.png', s1bfacts, legends=['5kb0kt'])
#
# s1msd, s1bfacts = m.load_sim_rmsds_from_file('5kb0kt.json')
# m.free_compare('5kb0kt.png', s1bfacts, legends=['5kb0kt'], title=)
#
# s1msd, s1bfacts = m.load_sim_rmsds_from_file('5kb0kt.json')
# m.free_compare('5kb0kt.png', s1bfacts, legends=['5kb0kt'])
# m.free_compare('15kb_hess_trial.png', bf, s2bfacts, ex_bfacts, legends=['ACT_15kb', 'sim_15kb', 'exp'], title='Sim vs. Hessian Prediction 15kb')
# m.free_compare('15kt_hess_trial.png', tf, s5bfacts, ex_bfacts, legends=['sim_15kt', 'ACT_15kt', 'exp'], title='Sim vs. Hessian Prediction 15kt')
# m.free_compare('15kbkt_hess_trial.png', bothf, s4bfacts, legends=['both_15', 'sim15kbkt'], title='Sim vs. Hessian Prediction 15kbkt')


# RC comparison on 1bu4
s1msd, s1b = m.load_sim_rmsds_from_file('10devs.json')
s2msd, s2b = m.load_sim_rmsds_from_file('7devs.json')
m.free_compare('rccomp.png', s1b, s2b, ex_bfacts, legends=['rc10', 'rc7', 'exp'])
m.free_compare('rccomp.png', s1b, ex_bfacts, legends=['rc10', 'exp'])






