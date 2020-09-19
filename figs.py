import models as m
import numpy as np

# 1bu4 soft matter fig wdir Gamma/1bu4
ccoord, cbfacts, normal_vectors = m.get_pdb_info('kdpg.pdb', returntype=4, orientations=True, multimodel=True)
fcoord = m.flatten(ccoord)
ex_bfacts = m.flatten(cbfacts)

#print(fcoord)
#print(ex_bfacts)

#GFP export


c = m.ANM(fcoord, ex_bfacts, cutoff=12, T=300)
c.calc_ANM_unitary()
print(c.ana_gamma)
c.anm_compare_bfactors('kdpg_bfacts.png')


a = m.ANMT(fcoord, normal_vectors, ex_bfacts, T=300)
a.ana_gamma = c.ana_gamma/0.05790
a.cutoff = 12
a.calc_a3s()

# m.export_to_simulation(a, 'kdpg.pdb', multimodel=True)

# print(a.ana_gamma)
# a.anm_compare_bfactors('1bu4ana.png')

s1msd, s1bfacts = m.load_sim_rmsds_from_file('act_kdpg.json')
m.free_compare('act_kdpg.png', ex_bfacts, c.ana_bfactors, s1bfacts, legends=['exp', 'anm', 'anmt'])


#s2msd, s2bfacts = m.load_sim_rmsds_from_file('bukbt3.json')
#m.free_compare('1bu4fig.png', c.ana_bfactors, s1bfacts, s2bfacts, ex_bfacts, legends=['ANM B-factors', 'Simulation ANM B-factors', 'Simulation ANMT B-factors', 'Experimental B-factors (from PDB)'])



# NSP3 softmatter fig wdir Gamma/NSP3
# ccoord, cbfacts = m.get_pdb_info('nsp3.pdb', returntype=2)
# fcoord = ccoord[0]
# ex_bfacts = cbfacts[0]
# a = m.ANM(fcoord, ex_bfacts, T=150, cutoff=18)
# a.calc_ANM_unitary()
# print(a.ana_gamma)
# s1msd, s1bfacts = m.load_sim_rmsds_from_file('dnsp.json')
# # s1bfact = [0.5 * x for x in s1bfacts]
# a_bfacts = [2.0 * x for x in a.ana_bfactors]
# m.free_compare('nsp3fig.png', a_bfacts, s1bfacts, ex_bfacts, legends=['Analytical B-factors', 'Simulation B-factors', 'Experimental B-factors (from PDB)'])


# GFP softmatter fig wdir Gamma
#ccoord, cbfacts, normal_vectors   = m.get_pdb_info('sgfp.pdb', returntype=4, orirntations=True)
#fcoord = ccoord[0]
#ex_bfacts = cbfacts[0]
#a = m.ANM(fcoord, ex_bfacts, T=300, cutoff=15)
#a.calc_ANM_unitary()
#print(a.ana_gamma)

# s1msd, s1bfacts = m.load_sim_rmsds_from_file('devsgfp4.954.json')
# s2msd, s2bfacts = m.load_sim_rmsds_from_file('dgk.json')
# #s1bfact = [0.5 * x for x in s1bfacts]
# a_bfacts = [1.0 * x for x in c.ana_bfactors]
# m.free_compare('gfpfig.png', a_bfacts, s1bfacts, s2bfacts, ex_bfacts, legends=['ANM B-factors', 'Simulation ANM B-factors', 'Simulation ANMT B-factors', 'Experimental B-factors'])


# Bending/ Torsional Fig wdir sims/ACT_1bu4/t1_10


#Create bond commands for chimera visualization of network model
# ccoord, cbfacts, normal_vectors = m.get_pdb_info('gfpisolate.pdb', orientations=True, returntype=4)
# fcoord = np.asarray(ccoord[0])
# ex_bfacts = cbfacts[0]
#
# cc = len(ex_bfacts)
# cutoff = 8
# bonds = []
# #missing residue 0, 1, 65, 66, 67
# for i in range(cc):
#     for j in range(cc):
#         if i >= j:
#             continue
#         else:
#             # Calculating distance
#             dx = fcoord[i, 0] - fcoord[j, 0]
#             dy = fcoord[i, 1] - fcoord[j, 1]
#             dz = fcoord[i, 2] - fcoord[j, 2]
#             dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
#             # too far, skips
#             if dist > float(cutoff):
#                 continue
#             else:
#                 # within cutoff added to bonds
#                 bonds.append((i, j))
#
# def check(i, j):
#     if i >= 63:
#         x = i + 2 + 3
#     else:
#         x = i + 2
#     if j >= 63:
#         y = j + 2 + 3
#     else:
#         y = j +2
#     return x, y
#
#
# for i, j in bonds:
#     x, y = check(i, j)
#     atom_i = '#:' + str(x) + '@CA'
#     atom_j = '#:' + str(y) + '@CA'
#     print('bond', atom_i, atom_j)




# m0b, b0b = m.load_sim_rmsds_from_file('pb0.json')
# m15b, b15b = m.load_sim_rmsds_from_file('pb15.json')
# m100b, b100b = m.load_sim_rmsds_from_file('pb100.json')
# m100kb, b100kb = m.load_sim_rmsds_from_file('pkb100.json')
# m100kt, b100kt = m.load_sim_rmsds_from_file('pb100kt.json')
# m1gb, b1gb = m.load_sim_rmsds_from_file('pb1g.json')
#
# m.free_compare('pep_fig.png', b0b, b15b, b100b, b1gb, b100kb, b100kt, legends=['both=0', 'both=15', 'both=100', 'both=1g', 'kb=100', 'kt=100'])


# m0b, b0b = m.load_sim_rmsds_from_file('10devs0.json')
#
# m25kb, b25kb = m.load_sim_rmsds_from_file('10devs25kb.json')
# m1gkb, b1gkb = m.load_sim_rmsds_from_file('10devskb1000.json')
#
# m.free_compare('kb_fig.png', b0b, b25kb, b1gkb, legends=['both=0', 'kb=25', 'kb=1000'])
#
#
# m25kt, b25kt = m.load_sim_rmsds_from_file('10devs25kt.json')
# m1gkt, b1gkt = m.load_sim_rmsds_from_file('10devs1gkt.json')
#
# m.free_compare('kt_fig.png', b0b, b25kt, b1gkt, legends=['both=0', 'kt=25', 'kt=1000'])
#
#
# m25b, b25b = m.load_sim_rmsds_from_file('10devs25.json')
# m15b, b15b = m.load_sim_rmsds_from_file('10devs15.json')
# m150b, b150b = m.load_sim_rmsds_from_file('10devs150.json')
# m1gb, b1gb = m.load_sim_rmsds_from_file('10devs150.json')
# m10gb, b10gb = m.load_sim_rmsds_from_file('10devs10g.json')
#
#
# m.free_compare('bth_fig.png', b0b, b15b, b1gb, b10gb, legends=['0', '15', '1k', '10k'])

# Hess Checker (Think I need to define relationship of a1 and a3 vectors)
# sim5 = 5*0.05709
# sim7 = 7*0.05709
# sim15 = 15*0.05709
# sim25 = 25*0.05709
# ANMT = m.ANMT(fcoord, normal_vectors, ex_bfacts, T=300)
# ANMT.calc_a3s()
# hkb= ANMT.calc_bend_hess(kb=sim25, kt=0)
# hkt = ANMT.calc_bend_hess(kb=0, kt=sim25)
# hb = ANMT.calc_bend_hess(kb=sim25, kt=sim25)
#
# #Normal ANM component
# anm = m.ANM(fcoord, ex_bfacts, cutoff=10, T=300)
# anm.calc_dist_matrix()
# ah = anm.calc_hess_fast_unitary(gamma=sim7)
# ih = anm.calc_inv_Hess(ah, cuda=False)
# #
# thkb = ANMT.total_hess(hkb, ah)
# thkt = ANMT.total_hess(hkt, ah)
# thb = ANMT.total_hess(hb, ah)
#
# ihkb = ANMT.calc_inv_Hess(thkb, cuda=False)
# ihkt = ANMT.calc_inv_Hess(thkt, cuda=False)
# ihb = ANMT.calc_inv_Hess(thb, cuda=False)
#
# bkb = ANMT.calc_bfacts(ihkb)
# bkt = ANMT.calc_bfacts(ihkt)
# bb = ANMT.calc_bfacts(ihb)
# # print(bf)
# # print(tf)
#
# m.free_compare('25kb.png', b25kb, bkb, legends=['sim', 'hess'])
# m.free_compare('25kt.png', b25kt, bkt, legends=['sim', 'hess'])
# m.free_compare('25b.png', b25b, bb, legends=['sim', 'hess'])
