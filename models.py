import Bio
import Bio.PDB
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
import sys
import os


# Helper Functions
def dist(coord, i, j):
    dx = coord[i, 0] - coord[j, 0]
    dy = coord[i, 1] - coord[j, 1]
    dz = coord[i, 2] - coord[j, 2]
    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def flatten(l1):
    return [item for sublist in l1 for item in sublist]

def divide_list(list, piecenum):
    divis = len(list) % piecenum
    if divis == 0:
        diff = len(list) // piecenum
    else:
        print("List is not divisble by the number of pieces you provided")
        sys.exit()

    for i in range(piecenum):
        if i == piecenum - 1:
            sublist = list[i * diff:]
            yield (sublist)
        else:
            sublist = list[i * diff:(i + 1) * diff]
            yield (sublist)


conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F',
        'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y',
        'MET': 'M'}


def get_pdb_info(pdb_file, returntype=1, multimodel='False'):
    if "/" in pdb_file:
        pdbid = pdb_file.rsplit('/', 1)[1].split('.')[0]
    else:
        pdbid = pdb_file.split('.')[0]
    structure = Bio.PDB.PDBParser().get_structure(pdbid, pdb_file)
    model = structure[0]
    if multimodel:
        model = Bio.PDB.Selection.unfold_entities(structure, 'C')
    chainids, chain_coords, chain_seqs, chain_bfactors = [], [], [], []
    # iterate through chains in pdb
    for chain in model:
        # Add chain
        chainids.append(chain.get_id())
        # empty coordinates each iteration to separate by chain
        coordtmp = []
        bfacttmp = []
        # iterate through residues
        for residue in chain.get_residues():
            tags = residue.get_full_id()
            # print(tags)
            if tags[3][0] == " ":
                # Get Residues one letter code
                onelettercode = conv[residue.get_resname()]
                # get residue number and identity per chain
                chain_seqs.append((tags[2], onelettercode))
                atoms = residue.get_atoms()
                for atom in atoms:
                    if atom.get_id() == 'CA':
                        coordinates = atom.get_coord()
                        bfactor = atom.get_bfactor()
                        bfacttmp.append(bfactor)
                        # Add coordinates to tmp container
                        coordtmp.append(coordinates)
        # Before next chain add all of this chain's coordinates to chain_coords
        chain_coords.append(coordtmp)
        chain_bfactors.append(bfacttmp)

    # The numbering on returntype is super arbitrary
    if returntype == 0:
        return chain_coords
    elif returntype == 1:
        return chain_coords, chainids, chain_seqs, chain_bfactors
    elif returntype == 2:
        return chain_coords, chain_bfactors
    elif returntype == 3:
        return chain_seqs

def free_compare(out, *data, legends=[]):
    cs = ['r', 'c', 'b']
    fig, ax = plt.subplots(1)
    for yid, y in enumerate(data):
        if legends:
            ax.plot(np.arange(0, len(y), 1), y, c=cs[yid % 3], label=legends[yid])
        else:
            ax.plot(np.arange(0, len(y), 1), y, c=cs[yid % 3])
    ax.legend()
    plt.savefig(out, dpi=600)
    plt.close()

def diff_sqrd(l1, l2):
    diff = 0
    for i in range(len(l1)):
        diff += abs(l1[i] ** 2 - l2[i] ** 2)
    return diff

def load_sim_rmsds_from_file(file):
    o = open(file, 'r')
    str_arr = o.read()
    o.close()
    # Removes beginning -> '{"RMSD (nm)"":
    arr1 = (str_arr.split(':')[1]).lstrip()
    # Removes End
    arr2 = arr1.split('}')[0]
    # Convert to List of Strings
    arr3 = arr2.strip('][]').split(', ')
    # Convert to List of Floats
    rmsds = [float(x) for x in arr3]
    sim_msds = [x ** 2 * 100 for x in rmsds]
    sim_bfactors = [(8 * math.pi ** 2) / 3 * x ** 2 * 100 for x in rmsds]
    return sim_msds, sim_bfactors




class ANM(object):

    def __init__(self, coord, exp_bfactors, T=300, cutoff=15):
        #Conversion factors for B Factors to Fluctiations
        self.bconv = (8. * math.pi ** 2) / 3.
        self.ibconv = 3./(8. * math.pi ** 2)
        #These are important for most loops
        self.sim_length = 8.518
        self.coord = coord
        self.cc = len(coord)
        #Data to match to
        self.exp_bfactors = exp_bfactors
        self.exp_msds = [self.ibconv*x for x in self.exp_bfactors]

        self.ana_bfactors = []
        self.msds = []
        self.ana_gamma = 0.
        # Angstroms
        self.cutoff = cutoff
        #Angstroms in 1 sim unit length
        self.sim_force_const = .05709  # (Sim Units to pN/A)
        # IN picoNetwtons/ Angstroms
        self.kb = 0.00138064852
        #Kelvin
        self.T = T

        self.distance_matrix = []
        # CUDA SPECIFICS
        self.cuda_initialized = False
        self.model_id = 'ANM'

    def calc_dist_matrix(self):
        d_matrix = np.full((self.cc, 4*self.cc), 0.0, dtype=np.float32)
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                else:
                    #Calculating distance
                    dx = self.coord[i,0]- self.coord[j,0]
                    dy = self.coord[i,1]- self.coord[j,1]
                    dz = self.coord[i,2]- self.coord[j,2]
                    dist = np.sqrt(dx**2 + dy**2 +dz**2)
                    # too far, skips
                    if dist > float(self.cutoff):
                        continue
                    else:
                        d_matrix[i, 4*j+1] = dx
                        d_matrix[i, 4*j+2] = dy
                        d_matrix[i, 4*j+3] = dz
                        d_matrix[i, 4*j] = dist
        self.distance_matrix = d_matrix

    def calc_hess_fast_sc(self, spring_constant_matrix):
        threeN = 3 * self.cc
        hess = np.zeros((threeN, threeN), dtype=np.float32)
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                di = self.distance_matrix[i, 4 * j:4 * j + 4]
                # Filter so that Hessian is only created for those bonds in bonds array
                if di[0] != 0:
                    di2 = np.square(di)
                    g = spring_constant_matrix[i, j]

                    diag = g * di2[1:4] / di2[0]

                    xy = g * (di[1] * di[2]) / di2[0]
                    xz = g * (di[1] * di[3]) / di2[0]
                    yz = g * (di[2] * di[3]) / di2[0]

                    full = np.asarray([[diag[0], xy, xz], [xy, diag[1], yz], [xz, yz, diag[2]]], order='F')

                    # Hii and Hjj
                    hess[3 * i:3 * i + 3, 3 * i:3 * i + 3] += full
                    hess[3 * j: 3 * j + 3, 3 * j:3 * j + 3] += full

                    # Hij and Hji
                    hess[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] -= full
                    hess[3 * j: 3 * j + 3, 3 * i: 3 * i + 3] -= full
        return hess

    def calc_hess_fast_unitary(self):
        threeN = 3 * self.cc
        hess = np.zeros((threeN, threeN), dtype=np.float32)
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                di = self.distance_matrix[i, 4 * j:4 * j + 4]
                # Filter so that Hessian is only created for those bonds in bonds array
                if di[0] != 0:
                    di2 = np.square(di)
                    g = 1.

                    diag = g * di2[1:4] / di2[0]

                    xy = g * (di[1] * di[2]) / di2[0]
                    xz = g * (di[1] * di[3]) / di2[0]
                    yz = g * (di[2] * di[3]) / di2[0]

                    full = np.asarray([[diag[0], xy, xz], [xy, diag[1], yz], [xz, yz, diag[2]]], order='F')

                    # Hii and Hjj
                    hess[3 * i:3 * i + 3, 3 * i:3 * i + 3] += full
                    hess[3 * j: 3 * j + 3, 3 * j:3 * j + 3] += full

                    # cHij and Hji
                    hess[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] -= full
                    hess[3 * j: 3 * j + 3, 3 * i: 3 * i + 3] -= full
        return hess

    def init_cuda(self):
        try:
            from pycuda import autoinit
            from pycuda import gpuarray
            import pycuda.driver as cuda
            import skcuda.linalg as cuda_la
            import skcuda.misc as cuda_misc
            cuda_misc.init()
            cuda_la.init()
        except ImportError:
            print('CUDA Initialization Failed. Check Dependencies')
            sys.exit()
        self.cuda_initialized = True

    def calc_inv_Hess(self, hess, cuda=False):
        if cuda:
            if self.cuda_initialized:
                pass
            else:
                self.init_cuda()
            thess = np.asarray(hess, dtype=np.float32, order='C')
            cu_hess = gpuarray.to_gpu(thess)
            u_gpu, s_gpu, vh_gpu = cuda_la.svd(cu_hess, 'S', 'S')
            U, w, Vt = u_gpu.get(), s_gpu.get(), vh_gpu.get()
        else:
            U, w, Vt = scipy.linalg.svd(hess, full_matrices=False)
        print(w)
        S = scipy.linalg.diagsvd(w, len(w), len(w))
        tol = 1e-4
        singular = w < tol
        invw = 1 / w
        invw[singular] = 0.
        hessinv = np.dot(np.dot(U, np.diag(invw)), Vt)
        return hessinv

    def save_inverse_Hessian(self, invhess, outfile):
        np.save(outfile, invhess)

    def load_inverse_Hessian(self, infile):
        invhess = np.load(infile)
        return invhess

    #finds closest particles to the coordindx
    # def find_nearest_particles(self, coordindx, cutoff):
    #
    #
    # def anm_remove_peak(self):
    #     hpeak = np.argmax(np.asarray(self.msds))

    def ANM_search(self, br, er, step):
        r = []
        sc_range = np.arange(br, er, step)
        for i in sc_range:
            g = float(i)
            r.append(diff_sqrd([x * 1. / g for x in self.msds], self.exp_msds))
        results = np.array(r)
        bg = np.argmin(results)
        return bg * step + br

    def ANM_fit_to_exp(self, start=0.001, end=5.0, step=0.001):
        g1 = self.ANM_search(start, end, step)
        self.ana_gamma = g1
        self.ana_msd = [x * 1 / self.ana_gamma for x in self.msds]
        self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x * 1 / self.ana_gamma for x in self.msds]

    def calc_msds(self, invhess):
        self.msds = []
        for i in range(self.cc):
            self.msds.append(self.kb * self.T * (invhess[3 * i, 3 * i] + invhess[3 * i + 1, 3 * i + 1] +
                                                 invhess[3 * i + 2, 3 * i + 2]))

    def anm_calc_bfactors(self):
        self.ana_bfactors = []
        self.ana_bfactors = [self.bconv*x*1/self.ana_gamma for x in self.msds]

    def calc_ANM_sc(self, spring_constant_matrix, cuda=False):
        self.calc_dist_matrix()
        hess = self.calc_hess_fast_sc(spring_constant_matrix)
        iH = self.calc_inv_Hess(hess, cuda=cuda)
        self.calc_msds(iH)
        self.anm_calc_bfactors()

    def calc_ANM_unitary(self, cuda=False):
        self.calc_dist_matrix()
        hess = self.calc_hess_fast_unitary()
        iH = self.calc_inv_Hess(hess, cuda=cuda)

        self.calc_msds(iH)
        print(self.msds)
        self.ANM_fit_to_exp()

    def anm_compare_bfactors(self, outfile):
        if self.ana_bfactors:
            free_compare(outfile, self.exp_bfactors, self.ana_bfactors,
                         legends=['Experimental  (PDB)', 'Analytical (ANM)' + str(round(self.ana_gamma*100, 3))+ "(pN/A)"])
        else:
            print('Analytical B Factors have not been Calculated')


#Implemented this paper's idea https://pubs-rsc-org.ezproxy1.lib.asu.edu/en/content/articlepdf/2018/cp/c7cp07177a
#Very Basic Implementation for C-A coarse graining
class MVPANM(ANM):
    def __init__(self, coord, exp_bfactors, cutoff=15, scale_resolution=15, k_factor=3, algorithim='ge', T=300):
        super().__init__(coord, exp_bfactors, T=T, cutoff=cutoff)
        # weight factor might change later we'll see
        self.w = 1.
        self.scale_resolution = scale_resolution
        self.k_factor = k_factor
        self.alg = algorithim
        # self.cutoff = cutoff

        self.spring_constant_matrix = []
        # Rigidity Functions, only calculate once and unique to each system
        self.kernels = []
        self.mu = []
        self.mu_s = []
        self.model_id = 'MVP'

    def algorithim(self, dist):
        # Can choose between Generalized Exponential and Generalized Lorentz Function
        def gen_exp(dist):
            return math.exp((-1. * dist / self.scale_resolution)) ** self.k_factor

        def gen_lor(dist):
            return 1. / (1. + (dist / self.scale_resolution) ** self.k_factor)

        if self.alg == 'ge':
            return gen_exp(dist)
        elif self.alg == 'gl':
            return gen_lor(dist)

    def mvp_compute_all_rigidity_functions(self):
        self.kernels = []
        self.mu = []
        self.mu_s = []
        for i in range(self.cc):
            ker_i = 0.
            for j in range(self.cc):
                d = dist(self.coord, i, j)
                if self.cutoff > 0. and d <= self.cutoff:
                    ker = self.algorithim(d)
                elif self.cutoff > 0. and d > self.cutoff:
                    ker = 0.
                else:
                    ker = self.algorithim(d)
                self.kernels.append(ker)
                ker_i += ker * self.w
            self.mu.append(ker_i)

        # replace ii with sum
        for i in range(self.cc):
            indx = i * self.cc + i
            self.kernels[indx] = -1 * self.mu[i]

        # Normalized density funciton
        mu_s = []
        min_mu = min(self.mu)
        max_mu = max(self.mu)
        for i in range(self.cc):
            mu_normed = (self.mu[i] - min_mu) / (max_mu - min_mu)
            mu_s.append(mu_normed)
        self.mu_s = mu_s

    def mvp_compute_gamma_1(self, i, j):
        return (1. + self.mu_s[i]) * (1. + self.mu_s[j])

    def mvp_compute_gamma_2(self, i, j):
        indx = i * self.cc + j
        return self.kernels[indx]

    def mvp_compute_spring_constants(self):
        if self.kernels and self.mu and self.mu_s:
            sc_matrix = np.full((self.cc, self.cc), 0.0)
            for i in range(self.cc):
                for j in range(self.cc):
                    if i == j:
                        spring_constant_ij = 1.
                    else:
                        spring_constant_ij = self.mvp_compute_gamma_1(i, j) * self.mvp_compute_gamma_2(i, j)
                    sc_matrix[i, j] = spring_constant_ij
            self.spring_constant_matrix = sc_matrix
        else:
            print('Must Compute Rigidity Functions Prior to Spring Constants')

    def simplify_matrix(self, percentile):
        cut_val = np.percentile(self.spring_constant_matrix, percentile)
        for i in range(self.cc):
            for j in range(self.cc):
                if self.spring_constant_matrix[i, j] < cut_val:
                    self.spring_constant_matrix[i, j] = 0

    def mvp_calc_bfactors(self, cuda=False):
        self.calc_dist_matrix()
        hess = self.calc_hess_fast_sc(self.spring_constant_matrix)
        iH = self.calc_inv_Hess(hess, cuda=cuda)
        self.calc_msds(iH)
        self.ana_bfactors = [self.bconv * x for x in self.msds]

    def mvp_fit_to_exp(self):
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            print('Check that sklearn module is installed')
            sys.exit()
        print(self.ana_bfactors)
        flex_data = np.asarray(self.ana_bfactors)
        exp_data = np.asarray(self.exp_bfactors)
        X = flex_data.reshape(-1, 1)
        Y = exp_data
        print(flex_data)
        fitting = LinearRegression(fit_intercept=False)
        fitting.fit(X, Y)
        slope = fitting.coef_
        self.spring_constant_matrix /= slope
        self.ana_bfactors *= slope

    def mvp_theor_bfactors(self, outfile):
        free_compare(outfile, self.exp_bfactors, self.ana_bfactors,
                         legends=['Experimental  (PDB)', 'Analytical (MVP)'])

    def calc_mvp(self, cuda=False):
        self.mvp_compute_all_rigidity_functions()
        self.mvp_compute_spring_constants()
        self.mvp_calc_bfactors(cuda=cuda)
        print(self.spring_constant_matrix)
        self.mvp_fit_to_exp()



class HANM(ANM):
    def __init__(self, coord, exp_bfactors, cutoff=15, T=300, scale_factor=0.3, mcycles=5, ncycles=7, cuda=False):
        super().__init__(coord, exp_bfactors, T=T, cutoff=cutoff)
        self.spring_constant_matrix = np.full((self.cc, self.cc), 1.)
        self.calc_ANM_unitary(cuda=cuda)
        self.spring_constant_matrix = np.full((self.cc, self.cc), self.ana_gamma)

        self.restraint_force_constants = []
        self.bond_fluctuations = []
        self.bond_fluctuations0 = []

        self.scale_factor = scale_factor
        self.mcycles = mcycles
        self.ncycles = ncycles
        self.routine_finished = False
        self.model_id = 'HANM'

    def hanm_calc_restraint_force_constant(self, bcal):
        restraint_force_constants = []
        for i in range(self.cc):
            ki_res = self.scale_factor * self.kb * self.T * 8 * math.pi ** 2. * (bcal[i] - self.exp_bfactors[i]) / \
                     (bcal[i] * self.exp_bfactors[i])
            restraint_force_constants.append(ki_res)
        # print(restraint_force_constants)
        return restraint_force_constants

    def hanm_add_restraints(self, hess, restraint_force_constants):
        for i in range(self.cc):
            hess[3 * i, 3 * i] += restraint_force_constants[i]
            hess[3 * i + 1, 3 * i + 1] += restraint_force_constants[i]
            hess[3 * i + 2, 3 * i + 2] += restraint_force_constants[i]

    def hanm_calc_bond_fluctuations(self, hess, cuda=False):
        if cuda:
            if not self.cuda_initialized:
                self.init_cuda()
        if cuda:
            thess = np.asarray(hess, dtype=np.float32, order='C')
            cu_hess = gpuarray.to_gpu(thess)
            cu_evecs, cu_evals = cuda_la.eig(cu_hess, imag='T')
            evals, evecs = cu_evals.get(), cu_evecs.get()
        else:
            evals, evecs = la.eig(hess)
        # print(self.evals)
        idx = np.argsort(abs(evals))
        evals = np.asarray(evals[idx])

        if cuda:
            evecs = np.asarray(evecs[idx, :])
            evecs = np.swapaxes(evecs, 1, 0)
        else:
            evecs = np.asarray(evecs[:, idx])

        # fig = plt.figure()
        # plt.imshow(evecs)
        # plt.savefig('evecs.png', dpi=600)

        # print('eVALS:', evals[6], evals[7])
        # print('evecs:', evecs[0, 6], evecs[0, 7], evecs[1, 6], evecs[1, 7])

        bcal = [0. for x in range(self.cc)]
        bond_fluc = np.full((self.cc, self.cc), 0.)
        for i in range(self.cc):
            for j in range(6, 3 * self.cc):
                if evals[j] != 0.:
                    bcal[i] += np.inner(evecs[3 * i: 3 * i + 3, j], evecs[3 * i: 3 * i + 3, j]) / evals[j]
            bcal[i] *= self.kb * self.T * self.bconv
        for i in range(self.cc):
            for j in range(self.cc):
                dis = self.distance_matrix[i, 4 * j]
                if dis:
                    tmp = np.asarray(self.distance_matrix[i, 4 * j + 1:4 * j + 4])
                    # print(tmp)
                    for k in range(6, 3 * self.cc):
                        p = tmp / dis * (evecs[3 * i:3 * i + 3, k] - evecs[3 * j:3 * j + 3, k])
                        if evals[k] != 0.:
                            bond_fluc[i, j] += np.sum(p) ** 2. / evals[k]
        return bcal, bond_fluc

    def hanm_nma(self, fc, fc_res, cuda=False):
        hess = self.calc_hess_fast_sc(fc)
        self.hanm_add_restraints(hess, fc_res)
        bcal, bond_fluc = self.hanm_calc_bond_fluctuations(hess, cuda=cuda)
        return bcal, bond_fluc

    def routine(self, cuda=False):

        mthreshold1 = 0.005  # relative
        mthreshold2 = 0.01  # absolute
        nthreshold = 0.001
        alpha = 1.0
        bcal = []
        bcalprev = [0. for x in range(self.cc)]

        fc_res0 = [0. for x in range(self.cc)]
        sc_mat_tmp = np.full((self.cc, self.cc), self.ana_gamma)

        if cuda:
            if not self.cuda_initialized:
                self.init_cuda()

        bcal, bond_fluc = self.hanm_nma(sc_mat_tmp, fc_res0, cuda=cuda)

        for i in range(self.mcycles):

            mcheck = 1
            for y in range(self.cc):
                rb1 = abs(bcal[y] - bcalprev[y]) / bcal[y]
                rb2 = abs(bcal[y] - self.exp_bfactors[y]) / self.exp_bfactors[y]

                # criterion can be changed if need be
                if rb2 > mthreshold2:
                    mcheck = 0.

            bcalprev = [x for x in bcal]

            if mcheck:
                self.ana_bfactors = bcal
                self.spring_constant_matrix = sc_mat_tmp
                self.routine_finished = True
                print("Bfactors have converged")
                break

            # Calc Restraint forces, add to hessian, calc bond_fluc, save as bond_fluctuations0
            fc_res = self.hanm_calc_restraint_force_constant(bcal)

            bcal, bond_fluc0 = self.hanm_nma(sc_mat_tmp, fc_res, cuda=cuda)


            for j in range(self.ncycles):

                bcal, bond_fluc = self.hanm_nma(sc_mat_tmp, fc_res0, cuda=cuda)

                ncheck = 1
                for x in range(self.cc):
                    for y in range(self.cc):
                        if x >= y:
                            continue
                        if self.distance_matrix[x, 4 * y] != 0.:
                            delta_fluc = bond_fluc[x, y] - bond_fluc0[x, y]
                            r2 = abs(delta_fluc) / bond_fluc0[x, y]
                            if r2 > nthreshold:
                                ncheck = 0.

                        # update Spring Constant Matrix
                        sc_mat_tmp[x, y] = 1. / ((1. / sc_mat_tmp[x, y]) - alpha * delta_fluc)
                        sc_mat_tmp[y, x] = sc_mat_tmp[x, y]

                if ncheck:
                    print("Force Constants have converged after %d mcycles and %d ncycles" % (i, j))
                    self.ana_bfactors = bcal
                    self.spring_constant_matrix = sc_mat_tmp
                    self.routine_finished = True
                    break
                else:
                    print("Force Constants have not converged after %d mcycles and %d ncycles" % (i, j))
        self.ana_bfactors = bcal
        self.spring_constant_matrix = sc_mat_tmp
        self.routine_finished = True

    def hanm_theor_bfactors(self, outfile):
        if self.routine_finished:
            free_compare(outfile, self.exp_bfactors, self.ana_bfactors,
                         legends=['Experimental  (PDB)', 'Analytical (HANM)'])
        else:
            print('HANM has not been run')


#Helper function for getting coordinates
def find_nearest(array,value):
    array=np.asarray(array)
    idx=(np.abs(array - value)).argmin()
    return array[idx]
#Easier to write
s=' '
n='\n'

class protein:
    def __init__(self, pdbfile, cutoff=15, pottype='s', potential=5.0, offset_indx=0, strand_offset=0,
                 diff_chains=True, import_sc=False, spring_constant_matrix=[]):
        self.su = 8.518
        self.boxsize = 0
        self.diff_chains = diff_chains
        self.pi = 0

        if "/" in pdbfile:
            pdbid = pdbfile.rsplit('/', 1)[1].split('.')[0]
        else:
            pdbid = pdbfile.split('.')[0]

        wdir = os.getcwd()
        # Outfiles they go into directory you call script from
        self.parfile = wdir + os.path.join('/generated.par')
        self.topfile = wdir + os.path.join('/generated.top')
        self.datfile = wdir + os.path.join('/generated.dat')

        self.sim_force_const = .05709
        self.pottype = pottype
        self.import_sc = import_sc
        if import_sc:
            self.spring_constant_matrix = spring_constant_matrix
            self.potential = 0.
        else:
            self.spring_constant_matrix = []
            self.potential = potential

        self.rc = cutoff
        self.topbonds = []
        self.chainnum = 0

        self.strand_offset = strand_offset
        self.offset_indx = offset_indx
        self.topology = []
        self.conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T',
                     'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A',
                     'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

        self.getalphacarbons(pdbid, pdbfile)

    def three_to_one_seq(self, res):
        ol = self.conv[res]
        return ol

    def adjust_coord(self, column, min):
        for i in range(0, self.pi):
            if min > 0:
                self.coord[i, column] -= min
            else:
                self.coord[i, column] += min

    def getalphacarbons(self, pdbid, pdbfile):
        structure = Bio.PDB.PDBParser().get_structure(pdbid, pdbfile)
        model = structure[0]
        # chain index, residue index, residue identitiy, CA coordinates
        cindx, rindx, rids, coord = [], [], [], []
        if len(model) == 1:
            self.single_chain = True
        else:
            self.single_chain = False
        for cid, chain in enumerate(model):
            for rid, residue in enumerate(chain.get_residues()):
                tags = residue.get_full_id()
                if tags[3][0] == " ":
                    onelettercode = self.three_to_one_seq(residue.get_resname())
                    atoms = residue.get_atoms()
                    for atom in atoms:
                        if atom.get_id() == 'CA':
                            coordinates = atom.get_coord()
                            cindx.append(cid)
                            rindx.append(rid)
                            rids.append(onelettercode)
                            coord.append(coordinates)
        self.topology = zip(cindx, rindx, rids)
        acs = np.divide(np.array(coord), self.su)
        self.coord = acs

    def WriteParFile(self, pottype='s', potential=5.0):
        make = open(self.parfile, 'w')
        self.pi = len(self.coord)  # Particle Index for range operations
        print(self.coord.shape, len(self.coord))
        print(self.pi)
        make.write(str(len(self.coord)))
        make.write(n)
        make.close()
        p = open(self.parfile, "a")
        for i in range(0, self.pi):
            for j in range(0, self.pi):
                if i >= j:
                    continue
                else:
                    dx = self.coord[i, 0] - self.coord[j, 0]
                    dy = self.coord[i, 1] - self.coord[j, 1]
                    dz = self.coord[i, 2] - self.coord[j, 2]
                    dist = math.sqrt(abs(dx) ** 2 + abs(dy) ** 2 + abs(dz) ** 2)
                    if self.pottype == 's':
                        if dist < (self.rc / self.su):
                            self.topbonds.append((i, j))
                            if abs(i - j) == 1:
                                if self.import_sc:
                                    spring_constant = self.spring_constant_matrix[i, j] / self.sim_force_const
                                    print(i + self.offset_indx, j + self.offset_indx, dist, self.pottype, spring_constant,
                                          file=p)
                                else:
                                    print(i + self.offset_indx, j + self.offset_indx, dist, self.pottype,
                                          self.potential, file=p)
                            else:
                                if self.import_sc:
                                    spring_constant = self.spring_constant_matrix[i, j] / self.sim_force_const
                                    print(i + self.offset_indx, j + self.offset_indx, dist, self.pottype, spring_constant,
                                          file=p)
                                else:
                                    print(i + self.offset_indx, j + self.offset_indx, dist, self.pottype, self.potential,
                                          file=p)
        p.close()

    def WriteConfFile(self):
        xcoord = np.array(self.coord[:-1, 0])
        ycoord = np.array(self.coord[:-1, 1])
        zcoord = np.array(self.coord[:-1, 2])
        xmin = find_nearest(xcoord, 0)
        ymin = find_nearest(ycoord, 0)
        zmin = find_nearest(zcoord, 0)
        self.adjust_coord(0, xmin)
        self.adjust_coord(1, ymin)
        self.adjust_coord(2, zmin)
        span = np.array(
            [np.max(xcoord) - np.min(xcoord), np.max(ycoord) - np.min(ycoord), np.max(zcoord) - np.min(zcoord)])
        self.boxsize = 2.5 * math.ceil(np.max(span))
        conf = open(self.datfile, "w")
        print('t = 0', file=conf)
        print("b =", str(self.boxsize), str(self.boxsize), str(self.boxsize), file=conf)
        print("E =", 0, 0, 0, file=conf)
        for i in range(0, self.pi):
            print(self.coord[i, 0], self.coord[i, 1], self.coord[i, 2], 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, file=conf)
        conf.close()

    def WriteTopFile(self):
        cindx, rindx, rids = zip(*self.topology)
        # Finding where the chains start and stop
        t_aft = [cindx.index(x) for x in np.arange(1, cindx[-1] + 1, 1)]
        t_pri = [x - 1 for x in t_aft]

        t = open(self.topfile, 'w')
        if self.diff_chains:
            print(self.pi, len(t_aft) + 1, file=t)
        else:
            print(self.pi, 1, file=t)
        if self.pottype == 's':
            # Get bonds
            fullneighs = []
            for j in range(0, self.pi):
                neighs = []
                for x, y in self.topbonds:
                    if x + 1 != y and x == j:
                        neighs.append(y)
                nebs = list(neighs)
                nebs_adj = [x + self.offset_indx for x in nebs]
                fullneighs.append(nebs_adj)

            if not self.diff_chains:
                t_aft, t_pri = [], []
                cindx = [cindx[0] for x in cindx]
            for cid, i in enumerate(cindx):
                ci_adj = -1 * (i + self.strand_offset + 1)
                rindx_adj = cid + self.offset_indx
                olc = rids[cid]
                bonds = fullneighs[cid]
                print(bonds)
                if self.pottype == 's':
                    if cid == self.pi - 1 or cid in t_pri:
                        print(ci_adj, olc, rindx_adj - 1, -1, *bonds, file=t)
                    elif cid == 0 or cid in t_aft:
                        print(ci_adj, olc, -1, rindx_adj + 1, *bonds, file=t)
                    else:
                        print(ci_adj, olc, rindx_adj - 1, rindx_adj + 1, *bonds, file=t)
        t.close()

    def WriteSimFiles(self):
        self.WriteParFile()
        self.WriteConfFile()
        self.WriteTopFile()



def export_to_simulation(model, pdbfile):
    if model.model_id == 'ANM':
        p = protein(pdbfile, cutoff=model.cutoff, potential=model.ana_gamma)
        p.WriteSimFiles()
    elif model.model_id == 'HANM' or model.model_id == 'MVP':
        p = protein(pdbfile, cutoff=model.cutoff, import_sc=True, spring_constant_matrix=model.spring_constant_matrix)
        p.WriteSimFiles()
