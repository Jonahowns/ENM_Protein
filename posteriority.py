import Bio
import Bio.PDB
import scipy.linalg
import matplotlib.pyplot as plt
import numpy
import numpy.linalg as la
import math
import statistics as stats
import numpy as np
import multiprocessing as mp
import sys
import os
import copy
from sklearn.linear_model import LinearRegression

# CUDA Dependencies
from pycuda import autoinit
from pycuda import gpuarray
import pycuda.driver as cuda
import skcuda.linalg as cuda_la
import skcuda.misc as cuda_misc

from numba import jit


@jit(nopython=True)
# Helper Functions

# Get distance between two vectors in coord matrix
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


def get_pdb_info(pdb_code, pdb_filename, returntype=1):
    structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
    model = structure[0]

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
            if tags[3][0] == " ":
                # Get Residues one letter code
                onelettercode = conv[residue.get_resname()]
                # get residue number and identity per chain
                chain_seqs.append((tags[0], onelettercode))
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


def svd_inv(matrix):
    U, w, Vt = scipy.linalg.svd(matrix, full_matrices=False)
    S = scipy.linalg.diagsvd(w, len(w), len(w))
    tol = 1e-6
    singular = w < tol
    invw = 1 / w
    invw[singular] = 0.
    inv = numpy.dot(numpy.dot(U, numpy.diag(invw)), Vt)
    return inv


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


class Structure_Maker():
    def __init__(self, pdb_code, pdb_filename, temp, ptype='monomer', cutoff=13):
        # TODO add temp
        self.temp = temp
        self.cutoff = cutoff
        # 3 to 1 letter codes
        self.conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T',
                     'PHE': 'F', 'ASN': 'N',
                     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
                     'TYR': 'Y', 'MET': 'M'}
        # Use Biopython to handle pdb file
        chain_coords, chainids, chain_seqs, chain_bfactors = get_pdb_info(pdb_code, pdb_filename)

        # Bool for later
        mono = False
        if ptype == 'monomer':
            # Is a monomer
            mono = True
            # Flatten array and convert to numpy array
            full_coordinates = np.asarray(flatten(chain_coords))
            full_bfactors = flatten(chain_bfactors)
        elif ptype == 'dimer':
            # Make sure the chains are divisble by two
            ishalf = len(chainids) % 2
            if not ishalf:
                print("Odd number of chains for a dimer")
                sys.exit()
            # Separate lists into their respective
            c1 = chain_coords[:len(chain_coords) // 2]
            c2 = chain_coords[len(chain_coords) // 2:]
            # Flatten and make numpy arrays
            full_coordinates = [np.asarray(flatten(c1)), np.asarray(flatten(c2))]
            # Bfactors
            b1, b2 = divide_list(chain_bfactors, 2)
            full_bfactors = [flatten(b1), flatten(b2)]
        elif ptype == 'trimer':
            # Make sure the chains are divisble by two
            isthird = len(chainids) % 3
            if not isthird:
                print("Number of chains not divisible by three for a trimer")
                sys.exit()
            # Separate List into trimers
            c1 = chain_coords[:len(chain_coords) // 3]
            c2 = chain_coords[len(chain_coords) // 3:2 * len(chain_coords)]
            c3 = chain_coords[2 * len(chain_coords) // 3:]
            # Flattend and make numpy arrays
            full_coordinates = [np.asarray(flatten(c1)), np.asarray(flatten(c2)), np.asarray(flatten(c3))]
            # Bfactors
            b1, b2, b3 = divide_list(chain_bfactors, 3)
            full_bfactors = [flatten(b1), flatten(b2), flatten(b3)]
        else:
            print("Type", ptype, 'Not Supported')
            sys.exit()

        # Now we have an easy way to access modify and work with each piece individually
        self.Solvers = []
        if mono:
            self.Solvers.append(
                BfactorSolver(chainids[0], full_coordinates, self.temp, full_bfactors, load_inv_hess=False,
                              cutoff=self.cutoff, solve=True))
        else:
            for xid, x in enumerate(full_coordinates):
                self.Solvers.append(BfactorSolver(chainids[xid], x, self.temp, full_bfactors[xid], load_inv_hess=False,
                                                  cutoff=self.cutoff, solve=True))


class BfactorSolver():
    def __init__(self, id, coord, temp, exp_bfactors, load_inv_hess=False, cutoff=13, solve=True, use_cuda=False,
                 load_test_sys=False, modeltype='anm', import_spring_constants=[]):
        # For later just declaring members
        self.exp_msd = []
        self.sim_msds = []
        self.sim_bfactors = []
        self.ana_gamma = 0
        self.ana_bfactors = []
        self.ana_msd = []
        self.sim_gamma = 0
        self.rawmsds = []
        # Protein 3 to one letter codes
        self.conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T',
                     'PHE': 'F', 'ASN': 'N',
                     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
                     'TYR': 'Y', 'MET': 'M'}
        # Constants
        self.sim_length = 8.518
        self.sim_force_const = .05709  # (Sim Units to N/M)

        self.kb = 1.38064852 * 10 ** -3
        # These will need to be calculated
        self.bonds = []
        # Import experimental bfactors and solve for the experimental MSD
        self.exp_bfactors = exp_bfactors
        self.exp_msd = [(3 * x) / (8 * math.pi ** 2) for x in self.exp_bfactors]
        # set cutoff
        self.cutoff = cutoff
        # Our coordinates
        self.coord = coord
        # this isn't super necessary just nice number for indexing
        self.cc = len(coord)
        # Temperatures
        self.T = temp
        self.ana_T = temp
        # self.use_cuda = use_cuda
        self.coord = coord

        if modeltype == 'anm':
            self.model = 'anm'
            # Usually Solve
            if solve:
                if load_inv_hess == True:
                    self.blank_invHess = []
                    self.blank_hess = []
                    self.bonds = []
                    # Have to load it yourself by calling load_inverse_Hessian
                else:
                    # Calculate bonds from specified cutoff and the coordinates and fills self.bonds
                    self.calc_bonds()
                    # Calculates Hessian and Inverse Hessian of system without spring constant weighting
                    # This is equivalent to the formulation usually presented in ANM theory
                    # Greatly increases computational speed rather than resolving hessian and inverse each time
                    self.blank_hess = self.calc_blank_hess()
                    self.blank_invHess = self.inv_Hess()
                    # Calculates Mean Squared Deviations still with no spring constant weighting
                    self.calc_rawmsds()
                    # Finds Minimized Square Difference by varying spring constant between Experimental b factors and the
                    # analytical ones we just found
                    self.fit_to_exp()
                    # Calculates simulation spring constant just a quick multiplication
                    self.calc_sim_gamma()

        elif modeltype == 'mvp':
            self.model = 'mvp'
            self.spring_constant_matrix = import_spring_constants
            self.cutoff = cutoff
            self.calc_bonds()
            self.blank_hess = self.calc_blank_hess(spring_constants_known=True)
            self.blank_invHess = self.inv_Hess()
            self.calc_rawmsds()
            self.ana_msd = self.rawmsds
            self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x for x in self.ana_msd]
            self.fitting_coef = 0

    # These are very nice for loading and storing the inverse hessian so it doesn't
    # have to be calculated again, as the system size grows it get quite computationally intensive
    def save_inverse_Hessian(self, outfile):
        np.save(outfile, self.blank_invHess)

    def load_inverse_Hessian(self, infile):
        self.blank_invHess = np.load(infile)

    # Fill self.bonds so Hessian can be calculated
    def calc_bonds(self):
        # Iterates through all pairs with i >= j
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                else:
                    # Calculating distance
                    dx = self.coord[i, 0] - self.coord[j, 0]
                    dy = self.coord[i, 1] - self.coord[j, 1]
                    dz = self.coord[i, 2] - self.coord[j, 2]
                    dist = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    # too far, skips
                    if dist > float(self.cutoff):
                        continue
                    else:
                        # within cutoff added to bonds
                        self.bonds.append((i, j, dist))

    def calc_blank_hess(self, spring_constants_known=False):
        threeN = 3 * self.cc
        hess = numpy.zeros((threeN, threeN))
        for i in range(self.cc):
            for j in range(self.cc):
                if i == j:
                    continue
                dxij = self.coord[i, 0] - self.coord[j, 0]
                dyij = self.coord[i, 1] - self.coord[j, 1]
                dzij = self.coord[i, 2] - self.coord[j, 2]

                # Filter so that Hessian is only created for those bonds in bonds array
                for x, y, dist in self.bonds:
                    if i == x and j == y:
                        if spring_constants_known:
                            g = self.spring_constant_matrix[i, j]
                        else:
                            g = 1
                        # creation of Hii (Borrowed from dfi.py by Banu Ozkan)
                        hess[3 * i, 3 * i] += g * ((dxij * dxij)) / dist ** 2
                        hess[3 * i + 1, 3 * i + 1] += g * ((dyij * dyij)) / dist ** 2
                        hess[3 * i + 2, 3 * i + 2] += g * ((dzij * dzij)) / dist ** 2

                        hess[3 * i, 3 * i + 1] += g * ((dxij * dyij)) / dist ** 2
                        hess[3 * i, 3 * i + 2] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 1, 3 * i] += g * ((dyij * dxij)) / dist ** 2

                        hess[3 * i + 1, 3 * i + 2] += g * ((dyij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * i] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * i + 1] += g * ((dyij * dzij)) / dist ** 2

                        # creation of Hij
                        hess[3 * i, 3 * j] -= g * ((dxij * dxij)) / dist ** 2
                        hess[3 * i + 1, 3 * j + 1] -= g * ((dyij * dyij)) / dist ** 2
                        hess[3 * i + 2, 3 * j + 2] -= g * ((dzij * dzij)) / dist ** 2

                        hess[3 * i, 3 * j + 1] -= g * ((dxij * dyij)) / dist ** 2
                        hess[3 * i, 3 * j + 2] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 1, 3 * j] -= g * ((dyij * dxij)) / dist ** 2

                        hess[3 * i + 1, 3 * j + 2] -= g * ((dyij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * j] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * j + 1] -= g * ((dyij * dzij)) / dist ** 2

                        # Have to do the other way as well to create Hji and Hjj
                        # creation of Hjj
                        hess[3 * j, 3 * j] += g * ((dxij * dxij)) / dist ** 2
                        hess[3 * j + 1, 3 * j + 1] += g * ((dyij * dyij)) / dist ** 2
                        hess[3 * j + 2, 3 * j + 2] += g * ((dzij * dzij)) / dist ** 2

                        hess[3 * j, 3 * j + 1] += g * ((dxij * dyij)) / dist ** 2
                        hess[3 * j, 3 * j + 2] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 1, 3 * j] += g * ((dyij * dxij)) / dist ** 2

                        hess[3 * j + 1, 3 * j + 2] += g * ((dyij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * j] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * j + 1] += g * ((dyij * dzij)) / dist ** 2

                        # creation of Hji
                        hess[3 * j, 3 * i] -= g * ((dxij * dxij)) / dist ** 2
                        hess[3 * j + 1, 3 * i + 1] -= g * ((dyij * dyij)) / dist ** 2
                        hess[3 * j + 2, 3 * i + 2] -= g * ((dzij * dzij)) / dist ** 2

                        hess[3 * j, 3 * i + 1] -= g * ((dxij * dyij)) / dist ** 2
                        hess[3 * j, 3 * i + 2] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 1, 3 * i] -= g * ((dyij * dxij)) / dist ** 2

                        hess[3 * j + 1, 3 * i + 2] -= g * ((dyij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * i] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * i + 1] -= g * ((dyij * dzij)) / dist ** 2
        return hess

    def inv_Hess(self, sc=False):
        # Calculate Psuedo-Inverse using SVD
        # if self.use_cuda:
        #     U, w, Vt = cuda_la.svd(self.blank_hess, jobu='S', jobvt='S')
        # else:
        U, w, Vt = scipy.linalg.svd(self.blank_hess, full_matrices=False)
        S = scipy.linalg.diagsvd(w, len(w), len(w))
        tol = 1e-6
        singular = w < tol
        invw = 1 / w
        invw[singular] = 0.
        hessinv = numpy.dot(numpy.dot(U, numpy.diag(invw)), Vt)
        if sc:
            np.fill_diagonal(hessinv, 0.)
            for i in range(self.cc):
                for j in range(self.cc):
                    if self.spring_constant_matrix[i, j] != 0.:
                        hessinv[3 * i: 3 * i + 3, 3 * j: 3 * j + 3] /= self.spring_constant_matrix[i, j]
                        # hessinv[3*i: 3*i +3, 3*i: 3*i +3] -= hessinv[3*i: 3*i + 3, 3*j: 3*j + 3]
                        # hessinv[3*j: 3*j +3, 3*j: 3*j +3] -= hessinv[3*i: 3*i + 3, 3*j: 3*j + 3]
                    else:
                        continue
                        # hessinv[3*i: 3*i + 3, 3*j: 3*j + 3] *= np.full((3, 3), 0.)
            for i in range(self.cc):
                hessinv[3 * i: 3 * i + 3, 3 * i: 3 * i + 3] = np.sum(hessinv[3 * i: 3 * i + 3, :])
        return hessinv

    def diff_sqrd(self, l1, l2):
        diff = 0
        for i in range(len(l1)):
            diff += abs(l1[i] ** 2 - l2[i] ** 2)
        return diff

    def diff_sqrd_no_peaks(self, l1, l2):
        diff = 0
        for i in range(len(l1)):
            if l1[i] > 2 * l2[i]:
                diff += 0
            else:
                diff += abs(l1[i] ** 2 - l2[i] ** 2)
        return diff

    def diff_sqrd_no_peaks_minimizer(self, l1, l2, ignr_indxs):
        diff = 0
        for idd, i in enumerate(l1):
            if idd not in ignr_indxs:
                diff += abs(l1[idd] ** 2 - l2[idd] ** 2)
        return diff

    def search_no_peaks(self, br, er, step, ignr_indxs=[]):
        r = []
        sc = numpy.arange(br, er, step)
        for i in sc:
            g = float(i)
            if ignr_indxs:
                r.append(
                    self.diff_sqrd_no_peaks_minimizer([x * 1. / g for x in self.rawmsds], self.exp_msd, ignr_indxs))
            else:
                r.append(self.diff_sqrd_no_peaks([x * 1. / g for x in self.rawmsds], self.exp_msd))
        results = numpy.array(r)
        bg = numpy.argmin(results)
        return bg * step + br

    def search(self, br, er, step):
        r = []
        sc = numpy.arange(br, er, step)
        for i in sc:
            g = float(i)
            r.append(self.diff_sqrd([x * 1. / g for x in self.rawmsds], self.exp_msd))
        results = numpy.array(r)
        bg = numpy.argmin(results)
        return bg * step + br

    def calc_rawmsds(self):
        self.rawmsds = []
        for i in range(self.cc):
            self.rawmsds.append(self.kb * self.T * (
                        self.blank_invHess[3 * i, 3 * i] + self.blank_invHess[3 * i + 1, 3 * i + 1] +
                        self.blank_invHess[3 * i + 2, 3 * i + 2]))

    def load_sim_rmsds_from_file(self, file):
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
        self.sim_msds = [x ** 2 * 100 for x in rmsds]
        self.sim_bfactors = [(8 * math.pi ** 2) / 3 * x ** 2 * 100 for x in rmsds]

    def load_sim_rmsds(self, rmsds):
        self.sim_msds = [x ** 2 * 100 for x in rmsds]
        self.sim_bfactors = [(8 * math.pi ** 2) / 3 * x ** 2 * 100 for x in rmsds]

    def update_ana(self):
        self.calc_rawmsds()
        self.ana_msd = [x for x in self.rawmsds]
        self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x for x in self.rawmsds]

    def fit_to_exp(self):
        if self.model == 'anm':
            g1 = self.search(0.001, 10.0, 0.001)
            self.ana_gamma = g1
            self.ana_msd = [x * 1 / self.ana_gamma for x in self.rawmsds]
            self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x * 1 / self.ana_gamma for x in self.rawmsds]
        elif self.model == 'mvp':
            self.update_ana()
            flex_data = np.asarray(self.ana_bfactors)
            exp_data = np.asarray(self.exp_bfactors)
            X = flex_data.reshape(-1, 1)
            Y = exp_data
            fitting = LinearRegression(fit_intercept=False)
            fitting.fit(X, Y)
            slope = fitting.coef_
            self.blank_invHess *= slope
            self.spring_constant_matrix /= slope
            # self.blank_hess = self.calc_blank_hess(spring_constants_known=True)
            # self.blank_invHess = self.inv_Hess()
            self.fitting_coef = slope

    def fit_to_exp_nopeaks(self, bindxs):
        g1 = self.search_no_peaks(0.001, 2.0, 0.001, ignr_indxs=bindxs)
        self.ana_gamma = g1
        self.ana_msd = [x * 1 / self.ana_gamma for x in self.rawmsds]
        self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x * 1 / self.ana_gamma for x in self.rawmsds]

    def calc_sim_gamma(self):
        self.sim_gamma = self.ana_gamma / self.sim_force_const

    def extrapolate_rawmsd(self, NewT):
        extrapolated = []
        for x in self.rawmsds:
            extrapolated.append(x / self.T * NewT)
        self.rawmsds = extrapolated

    def extrapolate_exp_msd(self, NewT):
        extrapolated = []
        for x in self.exp_msd:
            extrapolated.append(x / self.T * NewT)
        self.exp_msd = extrapolated
        self.exp_bfactors = [(8 * math.pi ** 2) / 3 * x for x in self.exp_msd]

    def extrapolate_ana_msd(self, NewT):
        extrapolated = []
        for x in self.ana_msd:
            extrapolated.append(x / self.T * NewT)
        self.ana_msd = extrapolated
        self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x for x in self.ana_msd]

    def compare_all(self, outpath, type):
        if type == 'msd':
            sim = self.sim_msds
            ana = self.ana_msd
            exp = self.exp_msd
        elif type == 'b':
            sim = self.sim_bfactors
            ana = self.ana_bfactors
            exp = self.exp_bfactors
        x = numpy.arange(1, self.cc + 1)
        plt.suptitle("GFP B-Factor Fitting")
        plt.plot(x, sim, c='r', label='simulation g=' + str(round(self.sim_gamma, 3)) + '(simForce/simLength)')
        plt.plot(x, ana, c='b', label='analytical g=' + str(round(self.ana_gamma * 100, 3)) + "(pN/A)")
        plt.plot(x, exp, c='g', label='experimental')
        plt.xlabel('Residue')
        plt.ylabel('Mean Squared Deviation (A^2)')
        plt.legend()
        plt.savefig(outpath, dpi=400)
        plt.close()

    def compare_ana_vs_exp(self, outpath, type, customdomain=[]):
        if type == 'msd':
            ana = self.ana_msd
            exp = self.exp_msd
        elif type == 'b':
            ana = self.ana_bfactors
            exp = self.exp_bfactors
        if customdomain:
            x = np.arange(customdomain[0], customdomain[1])
            ana = self.ana_bfactors[customdomain[0]:customdomain[1] + 1]
            exp = self.exp_bfactors[customdomain[0]:customdomain[1] + 1]
        else:
            x = numpy.arange(1, self.cc + 1)
        plt.plot(x, ana, c='b', label='analytical g=' + str(round(self.ana_gamma * 100, 3)) + "(pN/A)")
        plt.plot(x, exp, c='g', label='experimental')
        plt.xlabel('Residue')
        plt.ylabel('Mean Squared Deviation (A^2)')
        plt.legend()
        plt.savefig(outpath, dpi=400)
        plt.close()


# Implemented this paper's idea https://pubs-rsc-org.ezproxy1.lib.asu.edu/en/content/articlepdf/2018/cp/c7cp07177a
# Very Basic Implementation for C-A coarse graining
class MVPANM():
    def __init__(self, pdb_code, pdb_filename, scale_resolution, k_factor, algorithim='mvp', temp=0):
        # weight factor is uniform
        self.w = 1.
        # Get coordinates from pdb
        coord, bfact = get_pdb_info(pdb_code, pdb_filename, returntype=2)
        self.coord, self.exp_bfactors = np.asarray(flatten(coord)), flatten(bfact)
        self.n = len(self.coord)
        self.scale_resolution = scale_resolution
        self.k_factor = k_factor

        self.kb = 1.38064852 * 10 ** -3
        self.bconv = (8. * math.pi ** 2) / 3.
        if temp > 0:
            self.T = temp

        self.spring_constant_matrix = []
        self.alg = algorithim
        self.ana_bfactors = []
        self.ih_trace = []

        self.fri_m = 0
        self.fri_b = 0

    def compute_all_rigidity_functions(self, cutoff=0):
        self.kernels = []
        self.mu = []
        for i in range(self.n):
            ker_i = 0.
            for j in range(self.n):
                d = dist(self.coord, i, j)
                if cutoff > 0. and d <= cutoff:
                    ker = self.algorithim(d)
                elif cutoff > 0. and d > cutoff:
                    ker = 0.
                else:
                    ker = self.algorithim(d)
                self.kernels.append(ker)
                ker_i += ker * self.w
            self.mu.append(ker_i)

        # replace ii with sum
        for i in range(self.n):
            indx = i * self.n + i
            self.kernels[indx] = -1 * self.mu[i]

        # Normalized density funciton
        mu_s = []
        min_mu = min(self.mu)
        max_mu = max(self.mu)
        for i in range(self.n):
            mu_normed = (self.mu[i] - min_mu) / (max_mu - min_mu)
            mu_s.append(mu_normed)
        self.mu_s = mu_s

    def normalize_kernels(self):
        x = self.kernels
        kmin = min(x)
        kmax = max(x)
        self.kernels = [(i - kmin) / (kmax - kmin) for i in x]

    def algorithim(self, dist):
        def gen_exp(dist):
            return math.exp((-1. * dist / self.scale_resolution)) ** self.k_factor

        def gen_lor(dist):
            return 1. / (1. + (dist / self.scale_resolution) ** self.k_factor)

        if self.alg == 'ge':
            return gen_exp(dist)
        elif self.alg == 'gl':
            return gen_lor(dist)

    def fFRI(self, cutoff):
        self.kernels = []
        self.mu = []
        for i in range(self.n):
            ker_i = 0
            for j in range(self.n):
                d = dist(self.coord, i, j)
                if d <= cutoff:
                    ker = self.algorithim(d)
                else:
                    ker = 0.
                self.kernels.append(ker)
                ker_i += ker * self.w
            self.mu.append(ker_i)

        # replace ii with sum
        for i in range(self.n):
            indx = i * self.n + i
            self.kernels[indx] = -1 * self.mu[i]

    def predict_bfactors_fFRI(self, cutoff, outpath=''):
        f_i = []
        self.fFRI(cutoff)
        for i in range(self.n):
            f_tmp = 1. / self.mu[i]
            f_i.append(f_tmp)

        flex_data = np.asarray(f_i)
        exp_data = np.asarray(self.exp_bfactors)
        X = flex_data.reshape(-1, 1)
        Y = exp_data
        fitting = LinearRegression()
        fitting.fit(X, Y)
        slope, intercept = fitting.coef_, fitting.intercept_
        self.fri_b = intercept
        self.fri_m = slope
        proj_bfactors = [f * slope + intercept for f in f_i]
        self.ana_bfactors = proj_bfactors
        if outpath:
            self.compare_ana_vs_exp(outpath)

    def compute_gamma_1(self, i, j):
        return (1. + self.mu_s[i]) * (1. + self.mu_s[j])

    def compute_gamma_2(self, i, j):
        indx = i * self.n + j
        return self.kernels[indx]

    def compute_spring_constants(self):
        sc_matrix = np.full((self.n, self.n), 0.0)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    spring_constant_ij = 1.
                else:
                    spring_constant_ij = self.compute_gamma_1(i, j) * self.compute_gamma_2(i, j)
                sc_matrix[i, j] = spring_constant_ij
        self.spring_constant_matrix = sc_matrix

    def simplify_matrix(self, percentile):
        cut_val = np.percentile(self.spring_constant_matrix, percentile)
        for i in range(self.n):
            for j in range(self.n):
                if self.spring_constant_matrix[i, j] < cut_val:
                    self.spring_constant_matrix[i, j] = 0

    def compare_ana_vs_exp(self, outpath, customdomain=[]):
        ana = self.ana_bfactors
        exp = self.exp_bfactors

        if customdomain:
            x = np.arange(customdomain[0], customdomain[1])
            ana = self.ana_bfactors[customdomain[0]:customdomain[1] + 1]
            exp = self.exp_bfactors[customdomain[0]:customdomain[1] + 1]
        else:
            x = numpy.arange(1, self.n + 1)
        plt.plot(x, ana, c='b', label='analytical mvp')
        plt.plot(x, exp, c='g', label='experimental')
        plt.xlabel('Residue')
        plt.ylabel('Mean Squared Deviation (A^2)')
        plt.legend()
        plt.savefig(outpath, dpi=400)
        plt.close()


class ANM(object):
    def __init__(self, coord, exp_bfactors, T=300, cutoff=10):
        self.coord = coord
        self.cc = len(coord)
        self.exp_bfactors = exp_bfactors
        self.exp_msds = [(3 * x) / (8 * math.pi ** 2) for x in self.exp_bfactors]
        self.ana_bfactors = []
        self.hess = []
        self.invhess = []
        self.msds = []
        self.ana_gamma = 0.
        # Angstroms
        self.cutoff = cutoff
        self.sim_length = 8.518
        self.sim_force_const = .05709  # (Sim Units to pN/A)
        # IN picoNetwtons/ Angstroms
        self.kb = 1.38064852 * 10 ** -3
        self.kbt = 2.49434192
        # Kelvin
        self.T = T
        self.bonds = []

    # Fill self.bonds so Hessian can be calculated
    def calc_bonds(self):
        self.bonds = []
        # Iterates through all pairs with i >= j
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                else:
                    # Calculating distance
                    dx = self.coord[i, 0] - self.coord[j, 0]
                    dy = self.coord[i, 1] - self.coord[j, 1]
                    dz = self.coord[i, 2] - self.coord[j, 2]
                    dist = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    # too far, skips
                    if dist > float(self.cutoff):
                        continue
                    else:
                        # within cutoff added to bonds
                        self.bonds.append((i, j, dist))

    def calc_hess(self, spring_constant_matrix=False):
        threeN = 3 * self.cc
        hess = np.zeros((threeN, threeN), dtype=np.float32)
        for i in range(self.cc):
            for j in range(self.cc):
                if i == j:
                    continue
                dxij = self.coord[i, 0] - self.coord[j, 0]
                dyij = self.coord[i, 1] - self.coord[j, 1]
                dzij = self.coord[i, 2] - self.coord[j, 2]

                # Filter so that Hessian is only created for those bonds in bonds array
                for x, y, dist in self.bonds:
                    if i == x and j == y:
                        if type(spring_constant_matrix) is bool:
                            g = 1.
                        else:
                            g = spring_constant_matrix[i, j]

                        hess[3 * i, 3 * i] += g * ((dxij * dxij)) / dist ** 2
                        hess[3 * i + 1, 3 * i + 1] += g * ((dyij * dyij)) / dist ** 2
                        hess[3 * i + 2, 3 * i + 2] += g * ((dzij * dzij)) / dist ** 2

                        hess[3 * i, 3 * i + 1] += g * ((dxij * dyij)) / dist ** 2
                        hess[3 * i, 3 * i + 2] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 1, 3 * i] += g * ((dyij * dxij)) / dist ** 2

                        hess[3 * i + 1, 3 * i + 2] += g * ((dyij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * i] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * i + 1] += g * ((dyij * dzij)) / dist ** 2

                        # creation of Hij
                        hess[3 * i, 3 * j] -= g * ((dxij * dxij)) / dist ** 2
                        hess[3 * i + 1, 3 * j + 1] -= g * ((dyij * dyij)) / dist ** 2
                        hess[3 * i + 2, 3 * j + 2] -= g * ((dzij * dzij)) / dist ** 2

                        hess[3 * i, 3 * j + 1] -= g * ((dxij * dyij)) / dist ** 2
                        hess[3 * i, 3 * j + 2] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 1, 3 * j] -= g * ((dyij * dxij)) / dist ** 2

                        hess[3 * i + 1, 3 * j + 2] -= g * ((dyij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * j] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * i + 2, 3 * j + 1] -= g * ((dyij * dzij)) / dist ** 2

                        # Have to do the other way as well to create Hji and Hjj
                        # creation of Hjj
                        hess[3 * j, 3 * j] += g * ((dxij * dxij)) / dist ** 2
                        hess[3 * j + 1, 3 * j + 1] += g * ((dyij * dyij)) / dist ** 2
                        hess[3 * j + 2, 3 * j + 2] += g * ((dzij * dzij)) / dist ** 2

                        hess[3 * j, 3 * j + 1] += g * ((dxij * dyij)) / dist ** 2
                        hess[3 * j, 3 * j + 2] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 1, 3 * j] += g * ((dyij * dxij)) / dist ** 2

                        hess[3 * j + 1, 3 * j + 2] += g * ((dyij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * j] += g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * j + 1] += g * ((dyij * dzij)) / dist ** 2

                        # creation of Hji
                        hess[3 * j, 3 * i] -= g * ((dxij * dxij)) / dist ** 2
                        hess[3 * j + 1, 3 * i + 1] -= g * ((dyij * dyij)) / dist ** 2
                        hess[3 * j + 2, 3 * i + 2] -= g * ((dzij * dzij)) / dist ** 2

                        hess[3 * j, 3 * i + 1] -= g * ((dxij * dyij)) / dist ** 2
                        hess[3 * j, 3 * i + 2] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 1, 3 * i] -= g * ((dyij * dxij)) / dist ** 2

                        hess[3 * j + 1, 3 * i + 2] -= g * ((dyij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * i] -= g * ((dxij * dzij)) / dist ** 2
                        hess[3 * j + 2, 3 * i + 1] -= g * ((dyij * dzij)) / dist ** 2
        return hess

    def calc_inv_Hess(self, sc=False):
        # Calculate Psuedo-Inverse using SVD
        # if self.use_cuda:
        # U, w, Vt = cuda_la.svd(self.blank_hess, jobu='S', jobvt='S')
        # else:
        U, w, Vt = scipy.linalg.svd(self.hess, full_matrices=False)
        S = scipy.linalg.diagsvd(w, len(w), len(w))
        tol = 1e-6
        singular = w < tol
        invw = 1 / w
        invw[singular] = 0.
        hessinv = numpy.dot(numpy.dot(U, numpy.diag(invw)), Vt)
        return hessinv

    def save_inverse_Hessian(self, outfile):
        np.save(outfile, self.invhess)

    def load_inverse_Hessian(self, infile):
        self.invhess = np.load(infile)

    def search(self, br, er, step):
        r = []
        sc = numpy.arange(br, er, step)
        for i in sc:
            g = float(i)
            r.append(diff_sqrd([x * 1. / g for x in self.msds], self.exp_msds))
        results = numpy.array(r)
        bg = numpy.argmin(results)
        return bg * step + br

    def fit_to_exp(self):
        g1 = self.search(0.001, 10.0, 0.001)
        self.ana_gamma = g1
        self.ana_msd = [x * 1 / self.ana_gamma for x in self.msds]
        self.ana_bfactors = [(8 * math.pi ** 2) / 3 * x * 1 / self.ana_gamma for x in self.msds]

    def calc_msds(self):
        self.msds = []
        for i in range(self.cc):
            self.msds.append(self.kb * self.T * (
                        self.invhess[3 * i, 3 * i] + self.invhess[3 * i + 1, 3 * i + 1] + self.invhess[
                    3 * i + 2, 3 * i + 2]))

    def calc_bfactors(self):
        return [(8. * math.pi ** 2.) / 3. * x for x in self.msds]

    def calc_ANM(self, spring_constant_matrix=False):
        self.calc_bonds()
        self.hess = self.calc_hess(spring_constant_matrix=spring_constant_matrix)
        print(self.hess[0, 0], self.hess[3, 0])
        self.invhess = self.calc_inv_Hess()
        print(self.invhess[0, 0], self.invhess[3, 0])
        self.calc_msds()
        print(self.msds)
        self.fit_to_exp()
        # self.ana_bfactors = self.calc_bfactors()


class HANM(ANM):
    def __init__(self, coord, exp_bfactors, T=300, cutoff=10, scale_factor=0.3, mcycles=5, ncycles=5):
        super().__init__(coord, exp_bfactors, T=T, cutoff=cutoff)
        self.spring_constant_matrix = np.full((self.cc, self.cc), 1.)
        self.calc_bonds()
        self.calc_ANM(spring_constant_matrix=self.spring_constant_matrix)

        self.distance_matrix = self.calc_dist_matrix()

        self.scale_factor = scale_factor
        self.restraint_force_constants = []
        self.bconv = (8. * math.pi ** 2.) / 3.
        self.bond_fluctuations = []
        self.bond_fluctuations0 = []
        self.mcycles = mcycles
        self.ncycles = ncycles

    def calc_dist_matrix(self):
        d_matrix = np.full((self.cc, 4 * self.cc), 0.0, dtype=np.float32)
        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                else:
                    # Calculating distance
                    dx = self.coord[i, 0] - self.coord[j, 0]
                    dy = self.coord[i, 1] - self.coord[j, 1]
                    dz = self.coord[i, 2] - self.coord[j, 2]
                    dist = numpy.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    # too far, skips
                    if dist > float(self.cutoff):
                        continue
                    else:
                        d_matrix[i, 4 * j + 1] = dx
                        d_matrix[i, 4 * j + 2] = dy
                        d_matrix[i, 4 * j + 3] = dz
                        d_matrix[i, 4 * j] = dist
        return d_matrix

    def calc_hess_fast(self, spring_constant_matrix):
        threeN = 3 * self.cc
        hess = np.zeros((threeN, threeN), dtype=np.float32)

        for i in range(self.cc):
            for j in range(self.cc):
                if i >= j:
                    continue
                di = self.distance_matrix[i, 4 * j:4 * j + 4]
                if di[0] != 0:
                    di2 = np.square(di)

                    # Filter so that Hessian is only created for those bonds in bonds array
                    g = spring_constant_matrix[i, j]

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

    def calc_restraint_force_constant(self, bcal):
        restraint_force_constants = []
        for i in range(self.cc):
            ki_res = self.scale_factor * self.kb * self.T * 8 * math.pi ** 2. * (bcal[i] - self.exp_bfactors[i]) / (
                        bcal[i] * self.exp_bfactors[i])
            restraint_force_constants.append(ki_res)
        # print(restraint_force_constants)
        return restraint_force_constants

    def add_restraints(self, hess, restraint_force_constants):
        for i in range(self.cc):
            hess[3 * i, 3 * i] += restraint_force_constants[i]
            hess[3 * i + 1, 3 * i + 1] += restraint_force_constants[i]
            hess[3 * i + 2, 3 * i + 2] += restraint_force_constants[i]

    def calc_bond_fluctuations(self, hess, cuda=True):
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
        # evecs = np.asarray(evecs[:,idx])

        if cuda:
            evecs = np.asarray(evecs[idx, :])
            evecs = np.swapaxes(evecs, 1, 0)
        else:
            evecs = np.asarray(evecs[:, idx])

        # fig = plt.figure()
        # plt.imshow(evecs)
        # plt.savefig('evecs.png', dpi=600)

        print('eVALS:', evals[6], evals[7])
        print('evecs:', evecs[0, 6], evecs[0, 7], evecs[1, 6], evecs[1, 7])

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

    def NMA(self, fc, fc_res, cuda=True):
        hess = self.calc_hess_fast(fc)
        self.add_restraints(hess, fc_res)
        bcal, bond_fluc = self.calc_bond_fluctuations(hess, cuda=cuda)
        return bcal, bond_fluc

    def routine(self, cuda=True):

        mthreshold1 = 0.005  # relative
        mthreshold2 = 0.01  # absolute
        nthreshold = 0.001
        alpha = 1.0
        bcal = []
        bcalprev = [0. for x in range(self.cc)]

        fc_res0 = [0. for x in range(self.cc)]
        sc_mat_tmp = np.full((self.cc, self.cc), self.ana_gamma)

        if cuda:
            cuda_misc.init()
            cuda_la.init()
            bcal, bond_fluc = self.NMA(sc_mat_tmp, fc_res0, cuda=True)
        else:
            bcal, bond_fluc = self.NMA(sc_mat_tmp, fc_res0, cuda=False)

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
                print("Bfactors have converged")
                break

            # Calc Restraint forces, add to hessian, calc bond_fluc, save as bond_fluctuations0
            fc_res = self.calc_restraint_force_constant(bcal)

            if cuda:
                bcal, bond_fluc0 = self.NMA(sc_mat_tmp, fc_res, cuda=True)
            else:
                bcal, bond_fluc0 = self.NMA(sc_mat_tmp, fc_res, cuda=False)

            for j in range(self.ncycles):

                if cuda:
                    bcal, bond_fluc = self.NMA(sc_mat_tmp, fc_res0, cuda=True)
                else:
                    bcal, bond_fluc = self.NMA(sc_mat_tmp, fc_res0, cuda=False)

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
                    break
                else:
                    print("Force Constants have not converged after %d mcycles and %d ncycles" % (i, j))
        self.ana_bfactors = bcal
        self.spring_constant_matrix = sc_mat_tmp


# Helper function for getting coorinates
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Easier to write
s = ' '
n = '\n'


# The offset_indx is the particle index this topology/parameter file to start from
# strand_offset is how many Protein strands before this one
# The diff_chains is for differentiating the chains in the pdb file,
# its there just in case but haven't needed it yet you'll want to keep this as False
class protein:
    def __init__(self, pdbfile, cutoff=10, pottype='s', potential=5.0, offset_indx=0, strand_offset=0,
                 diff_chains=False, backbone_k=0, neigh_radius=0, onionize=False, sc=False, spring_constant_matrix=[]):
        self.su = 8.518
        self.boxsize = 0
        self.diff_chains = diff_chains

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
        self.sc = sc
        if sc:
            self.spring_constant_matrix = spring_constant_matrix
            self.potential = 0.
        else:
            self.spring_constant_matrix = []
            self.potential = potential

        self.rc = cutoff
        self.onionize = onionize
        self.topbonds = []
        self.chainnum = 0
        self.backbone_k = backbone_k
        self.neigh_radius = neigh_radius
        self.strand_offset = strand_offset
        self.offset_indx = offset_indx
        self.topology = []
        self.conv = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ILE': 'I', 'PRO': 'P', 'THR': 'T',
                     'PHE': 'F', 'ASN': 'N',
                     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'ALA': 'A', 'VAL': 'V', 'GLU': 'E',
                     'TYR': 'Y', 'MET': 'M'}
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
        print(self.pi)
        make.write(str(len(self.coord)))
        make.write(n)
        make.close()
        p = open(self.parfile, "a")
        for i in range(0, self.pi):
            for j in range(0, self.pi):
                if self.neigh_radius > 0 and abs(i - j) > self.neigh_radius:
                    continue
                if i >= j:
                    continue
                else:
                    dx = self.coord[i, 0] - self.coord[j, 0]
                    dy = self.coord[i, 1] - self.coord[j, 1]
                    dz = self.coord[i, 2] - self.coord[j, 2]
                    dist = math.sqrt(abs(dx) ** 2 + abs(dy) ** 2 + abs(dz) ** 2)
                    if self.onionize:
                        diff = abs(i - j) - 1
                    else:
                        diff = 0
                    if pottype == 's':
                        if dist < (self.rc / self.su):
                            self.topbonds.append((i, j))
                            if abs(i - j) == 1:
                                if self.sc:
                                    spring_constant = self.spring_constant_matrix[i, j] / self.sim_force_const
                                    print(i + self.offset_indx, j + self.offset_indx, dist, pottype, spring_constant,
                                          file=p)
                                else:
                                    print(i + self.offset_indx, j + self.offset_indx, dist, pottype,
                                          potential + self.backbone_k, file=p)
                            else:
                                if self.sc:
                                    spring_constant = self.spring_constant_matrix[i, j] / self.sim_force_const
                                    print(i + self.offset_indx, j + self.offset_indx, dist, pottype, spring_constant,
                                          file=p)
                                else:
                                    print(i + self.offset_indx, j + self.offset_indx, dist, pottype, potential - diff,
                                          file=p)
                    if pottype == 'i' or pottype == 'e':
                        self.topbonds.append((i, j))
                        print(i + self.offset_indx, j + self.offset_indx, dist, pottype, potential, file=p)
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

    def WriteTopFile(self, pottype='s'):
        cindx, rindx, rids = zip(*self.topology)
        # Finding where the chains start and stop
        t_aft = [cindx.index(x) for x in np.arange(1, cindx[-1] + 1, 1)]
        t_pri = [x - 1 for x in t_aft]

        t = open(self.topfile, 'w')
        if self.diff_chains:
            print(self.pi, len(t_aft) + 1, file=t)
        else:
            print(self.pi, 1, file=t)
        if pottype == 's':
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
                if pottype == 's':
                    if cid == self.pi - 1 or cid in t_pri:
                        print(ci_adj, olc, rindx_adj - 1, -1, *bonds, file=t)
                    elif cid == 0 or cid in t_aft:
                        print(ci_adj, olc, -1, rindx_adj + 1, *bonds, file=t)
                    else:
                        print(ci_adj, olc, rindx_adj - 1, rindx_adj + 1, *bonds, file=t)
        t.close()

    def WriteSimFiles(self):
        if self.potential:
            self.WriteParFile(self.pottype, self.potential)
        else:
            self.WriteParFile(self.pottype, self.potential)
        self.WriteConfFile()
        self.WriteTopFile(self.pottype)


def write_protein(pdbfile, cutoff, model='anm', potential=0, pottype='s', offset_indx=0, strand_offset=0,
                  spring_constant_matrix=[], diff_chains=False, backbone_k=0, neigh_radius=0, onionize=False):
    if model == 'anm':
        p = protein(pdbfile, cutoff=cutoff, pottype=pottype, potential=potential, offset_indx=offset_indx,
                    strand_offset=strand_offset, diff_chains=diff_chains, backbone_k=backbone_k,
                    neigh_radius=neigh_radius, onionize=onionize)
    elif model == 'mvp':
        p = protein(pdbfile, cutoff=cutoff, pottype=pottype, potential=0., offset_indx=offset_indx,
                    strand_offset=strand_offset, diff_chains=diff_chains, sc=True,
                    spring_constant_matrix=spring_constant_matrix)
    p.WriteSimFiles()


def tip_check(pdbfile, outfile, c=15, t=300):
    if "/" in pdbfile:
        pdbid = pdbfile.rsplit('/', 1)[1].split('.')[0]
    else:
        pdbid = pdbfile.split('.')[0]
    coord, exp_bfacts = get_pdb_info(pdbid, pdbfile, returntype=2)
    fcoord = np.asarray(flatten(coord))
    ex_bfacts = flatten(exp_bfacts)
    check = ANM(fcoord, ex_bfacts, T=t, cutoff=c)
    check.calc_ANM(spring_constant_matrix=False)
    free_compare(outfile, check.ana_bfactors, check.exp_bfactors, legends=['ana', 'exp'])


tip_check('1bu4.pdb', '1bu4_check.png')

# HANM
# coord, exp_bfacts = get_pdb_info('histone', 'histone.pdb', returntype=2)
# # print(coord)
# fcoord = np.asarray(flatten(coord))
# # print(fcoord)
# # print(exp_bfacts)
# ex_bfacts = flatten(exp_bfacts)


# trial = MVPANM('histone', 'histone.pdb', 18, 3, algorithim='ge')
# trial.compute_all_rigidity_functions(cutoff=18)
# trial.compute_spring_constants()

# t1 = ANM(fcoord, ex_bfacts, T=300, cutoff=18)
# t1.calc_bonds()
# t1.calc_ANM(spring_constant_matrix=trial.spring_constant_matrix)


# bt2 = BfactorSolver ('j', trial.coord, 300, ex_bfacts, cutoff=18, solve=False, import_spring_constants=trial.spring_constant_matrix, modeltype='mvp')
# bt2.calc_bonds()
# bt2.blank_hess = bt2.calc_blank_hess(spring_constants_known=True)
# print(bt2.spring_constant_matrix[0,1])
# bt2.blank_invHess = bt2.inv_Hess()
# bt2.fit_to_exp()
# bt2.update_ana()
# print(bt2.spring_constant_matrix[0,1])
# nsc = bt2.spring_constant_matrix
# bt2.compare_ana_vs_exp('mvp20cutoff', 'b')


# trial = HANM(fcoord, ex_bfacts, mcycles=20, ncycles=10, cutoff=15, scale_factor=0.3)

# trial.routine(cuda=True)

# x = trial.spring_constant_matrix
# print(x)
# sm, sb = load_sim_rmsds_from_file('hanm.devs')

# THIS AIN"T IT CHIEF
# free_compare('mvp_hist_trial.png', trial.ana_bfactors, trial.exp_bfactors, legends=['ana', 'exp'])

# write_protein('histone.pdb', 15, model="mvp", spring_constant_matrix=trial.spring_constant_matrix)


############ Histone Model ###################
# write_protein('histone.pdb', 10, model='anm', potential=10, pottype='s', diff_chains=True)


# ############## LET"S MAKE THAT MVP MODEL WORK ############

# m1 = MVPANM('1bu4', '1bu4.pdb', 15, 2, algorithim='ge', temp=300)
# m1.compute_all_rigidity_functions(cutoff=20)
# m1.compute_spring_constants()
# bt2 = BfactorSolver ('j', m1.coord, 300, m1.exp_bfactors, cutoff=20, solve=False, import_spring_constants=m1.spring_constant_matrix, modeltype='mvp')
# bt2.calc_bonds()
# bt2.blank_hess = bt2.calc_blank_hess(spring_constants_known=True)
# print(bt2.spring_constant_matrix[0,1])
# bt2.blank_invHess = bt2.inv_Hess()
# bt2.fit_to_exp()
# bt2.update_ana()
# print(bt2.spring_constant_matrix[0,1])
# nsc = bt2.spring_constant_matrix
# bt2.compare_ana_vs_exp('mvp10cutoff', 'b')


# bt = BfactorSolver ('j', m1.coord, 300, m1.exp_bfactors, cutoff=20, solve=False, modeltype='mvp', import_spring_constants=nsc)
# bt.calc_bonds()
# bt.blank_hess = bt.calc_blank_hess(spring_constants_known=True)
# bt.blank_invHess = bt.inv_Hess()
# # #

# # # bt.blank_invHess = m1.spring_constant_matrix
# # # # bt.rawmsds = [m1.spring_constant_matrix[i, i] for i in range(m1.n)]
# # # print(bt.rawmsds)
# bt.update_ana()


# bt.load_sim_rmsds_from_file('mvp20devs.json')
# bt.sim_bfactors = [x for x in bt.sim_bfactors]
# # bt.ana_bfactors = [m1.kb * m1.T * x for x in bt.rawmsds]


# # # # print(m1.ana_bfactors[0], bt.ana_bfactors[0])
# # # # conv=[m1.ana_bfactors[i]/ bt.ana_bfactors[i] for i in range(len(m1.ana_bfactors))]
# # # # print(conv)

# # bt.compare_ana_vs_exp('mvpconfirm.png', 'b')
# bt.compare_all('mvpconfirm.png', 'b')

# # write_protein('1bu4.pdb', 20, model='mvp', spring_constant_matrix=nsc)

# # m1.check_spring_matrix()

# # bt.compare_ana_vs_exp('mvpfit.png', 'b')
# # print(sc)


#######################################################################################################
################# BELOW IS OLD FUNCTIONS/DATA THAT MAY BE OF SOME USE LATER SO I SAVED THEM ###########
#######################################################################################################
# tcoord = numpy.zeros((3,3))
# tcoord[0] = [0.0, 0.0, 0.0]
# tcoord[1] = [2.5, 0.0, 0.0]
# tcoord[2] = [1.25, 1.25*3.**(1./2.), 0.0]

# TEST SYTEM FOR BOND FEATURES
# if load_test_sys:
#     hess = numpy.zeros((9,9))
#     self.coord = tcoord
#     self.Hess_Bond_Edit(self.coord, hess, 0, 1)
#     self.Hess_Bond_Edit(self.coord, hess, 1, 2)
#     self.Hess_Bond_Edit(self.coord, hess, 0, 1)
#     self.blank_hess = hess
#     self.blank_invHess = np.asmatrix(self.inv_Hess())
#     numpy.savetxt('trial3.csv', self.blank_invHess, delimiter=',')

#  def Hess_Bond_Edit(self, coord, hess, i, j):
#         dxij=coord[i,0]-coord[j,0]
#         dyij=coord[i,1]-coord[j,1]
#         dzij=coord[i,2]-coord[j,2]

#         dist = numpy.sqrt(dxij**2 + dyij**2 +dzij**2)

#         hess[3*i,3*i] += ((dxij*dxij))/dist**2
#         hess[3*i+1,3*i+1] += ((dyij*dyij))/dist**2
#         hess[3*i+2,3*i+2] += ((dzij*dzij))/dist**2

#         hess[3*i,3*i+1] += ((dxij*dyij))/dist**2
#         hess[3*i,3*i+2] += ((dxij*dzij))/dist**2
#         hess[3*i+1,3*i] += ((dyij*dxij))/dist**2

#         hess[3*i+1,3*i+2] += ((dyij*dzij))/dist**2
#         hess[3*i+2,3*i] += ((dxij*dzij))/dist**2
#         hess[3*i+2,3*i+1] += ((dyij*dzij))/dist**2

#         hess[3*i,3*j] -= ((dxij*dxij))/dist**2
#         hess[3*i+1,3*j+1] -= ((dyij*dyij))/dist**2
#         hess[3*i+2,3*j+2] -= ((dzij*dzij))/dist**2

#         hess[3*i,3*j+1] -= ((dxij*dyij))/dist**2
#         hess[3*i,3*j+2] -= ((dxij*dzij))/dist**2
#         hess[3*i+1,3*j] -= ((dyij*dxij))/dist**2

#         hess[3*i+1,3*j+2] -= ((dyij*dzij))/dist**2
#         hess[3*i+2,3*j] -= ((dxij*dzij))/dist**2
#         hess[3*i+2,3*j+1] -= ((dyij*dzij))/dist**2

#         hess[3*j,3*j] += ((dxij*dxij))/dist**2
#         hess[3*j+1,3*j+1] += ((dyij*dyij))/dist**2
#         hess[3*j+2,3*j+2] += ((dzij*dzij))/dist**2

#         hess[3*j,3*j+1] += ((dxij*dyij))/dist**2
#         hess[3*j,3*j+2] += ((dxij*dzij))/dist**2
#         hess[3*j+1,3*j] += ((dyij*dxij))/dist**2

#         hess[3*j+1,3*j+2] += ((dyij*dzij))/dist**2
#         hess[3*j+2,3*j] += ((dxij*dzij))/dist**2
#         hess[3*j+2,3*j+1] += ((dyij*dzij))/dist**2

#         #creation of Hji
#         hess[3*j,3*i] -= ((dxij*dxij))/dist**2
#         hess[3*j+1,3*i+1] -= ((dyij*dyij))/dist**2
#         hess[3*j+2,3*i+2] -= ((dzij*dzij))/dist**2

#         hess[3*j,3*i+1] -= ((dxij*dyij))/dist**2
#         hess[3*j,3*i+2] -= ((dxij*dzij))/dist**2
#         hess[3*j+1,3*i] -= ((dyij*dxij))/dist**2

#         hess[3*j+1,3*i+2] -= ((dyij*dzij))/dist**2
#         hess[3*j+2,3*i] -= ((dxij*dzij))/dist**2
#         hess[3*j+2,3*i+1] -= ((dyij*dzij))/dist**2


# def compute_fFRI_spring_constants(self, cutoff, type='mvp', norm='False'):
#     if norm:
#         self.normalize_kernels()
#     self.fFRI(cutoff)
#     sc_matrix = np.full((self.n, self.n), 0.0)
#     for i in range(self.n):
#         for j in range(self.n):
#             g2 = self.compute_gamma_2(i, j)
#             if g2 != 0.:
#                 if type=='mvp':
#                     spring_constant_ij = self.compute_gamma_1(i, j) * self.compute_gamma_2(i, j)
#                 elif type=='ker':
#                     spring_constant_ij = self.compute_gamma_2(i, j)
#                 elif type=='invker':
#                     spring_constant_ij = -1./self.compute_gamma_2(i,j)
#                 elif type=='uni':
#                     spring_constant_ij = 1.
#                 elif type=='fij':
#                     spring_constant_ij = self.fri_m * -1./self.compute_gamma_2(i, j) + self.fri_b
#             else:
#                 spring_constant_ij = 0.
#             sc_matrix[i, j] = spring_constant_ij
#     self.spring_constant_matrix = sc_matrix


# def calc_anm_sc_matrix(self, invhess):
#     ih_const = (self.bconv *self.kb *self.T)
#     # iHt = np.asarray([(self.fri_m * 1./x + self.fri_b) * ih_const if x != 0. else 0. for x  in self.kernels])
#     # iHt = np.asarray([1./x * ih_const if x != 0. else 0. for x  in self.kernels])
#     # iHij = iHt.reshape(self.n, self.n)

#     ih_trace = np.full((self.n, self.n), 0.0)
#     for i in range(self.n):
#         for j in range(self.n):
#             ih_trace[i, j] = (invhess[3*i, 3*j] + invhess[3*i+1, 3*j+1] + invhess[3*i+2, 3*j+2]) * ih_const   #np.trace(invhess[3*i: 3*i +3, 3*j: 3*j+ 3])

#     self.ih_trace=ih_trace

#     # for i in range(self.n):
#     #     for j in range(self.n):
#     #         if self.spring_constant_matrix[i, j] == 0:
#     #             self.ih_trace[i, j] = 0.

#     # fih_trace = self.ih_trace.flatten()
#     # f_scm = self.spring_constant_matrix.flatten()

#     # final_scm = [f_scm[i]/]

#     spc = self.spring_constant_matrix

#     isc_matrix = np.divide(spc, ih_trace, where=spc!=0, out=np.zeros_like(spc)).flatten()
#     sc_matrix = np.asarray([1./x if x!=0. else 0. for x in isc_matrix])
#     fsc = sc_matrix.reshape(self.n, self.n)

#     # self.spring_constant_matrix = isc_matrix
#     self.spring_constant_matrix = fsc

#     # #n by n
#     # fullbmat = np.asarray([1./x * self.fri_m + self.fri_b  if x != 0 else 0. for x in self.kernels])
#     # #fi is our psuedo inverse hessian
#     # psuedo_iH = np.asarray([1./x if x != 0 else 0. for x in self.kernels])
#     # gih = np.divide(fullbmat, (8.*math.pi**2 / 3.)* self.kb * self.T)
#     # sc_trace = gih
#     # # Might comment this out
#     # # for i in range(self.n):
#     # #     for j in range(self.n):
#     # #         sc_trace[i, j] = np.trace(invhess[3*i: 3*i +3, 3*j: 3*j+ 3])


#     # self.sc_trace = sc_trace

#     # inv_sc = np.asarray([gih[i]/ psuedo_iH[i] if gih[i] != 0 else 0. for i in range(len(self.sc_trace))])

#     # inv_sc_matrix = inv_sc.reshape(self.n, self.n)

#     # def inv(x):
#     #     if x==0:
#     #         return 0.
#     #     else:
#     #         return 1./x
#     # sc = np.asarray([inv(x) for x in inv_sc])
#     # sc_matrix = sc.reshape(self.n, self.n)
#     # print(sc_matrix)
#     # self.spring_constant_matrix = sc_matrix


# burmsdnm = [0.13269182451500522, 0.08724412183387006, 0.07866917227179886, 0.06175685887482556, 0.0630303767978042, 0.05881155786966248, 0.06973489964016388, 0.07647490175130399, 0.0610442609339112, 0.0651263939577753, 0.0588044459848558, 0.0722399700754445, 0.07440572302544, 0.07351736458427033, 0.05934508128493908, 0.051696326660851645, 0.059819122439649985, 0.06115256444258076, 0.05303800502414082, 0.050761019955029546, 0.05913928722221178, 0.055134238548156765, 0.04919280984523758, 0.05340794172519169, 0.06312326445331512, 0.06194558453863041, 0.06310006860319836, 0.08293232652844523, 0.10259091100232934, 0.09311704429510057, 0.09112780262132783, 0.07107709846700765, 0.07031452499821603, 0.07796425243603111, 0.09520164001572708, 0.08087651499242346, 0.07349472585485685, 0.05540059785117104, 0.05250293434778304, 0.06070979849047541, 0.06784150095846164, 0.05727463019072801, 0.07324568471229655, 0.06243220513881922, 0.08680699814575565, 0.07921034453260149, 0.08423964659866785, 0.07205207094987465, 0.08493332385784695, 0.06944494709172926, 0.08922887440047844, 0.08158830132364261, 0.07512945251271584, 0.06786164424340618, 0.07040514002899924, 0.055189015092387175, 0.05152701600271178, 0.04852017626922906, 0.04730950591184101, 0.04709792290956115, 0.047378304905112904, 0.053419830183369314, 0.060184063152508904, 0.07500328219220145, 0.07007352905730778, 0.0700586503758358, 0.05887387441784303, 0.055628048819965095, 0.06594788325705074, 0.07344472086047212, 0.08377718447528255, 0.0791817498079086, 0.06138173871599588, 0.06172473880753296, 0.061144416253766945, 0.05188711672011853, 0.04686450677942461, 0.04621644387041831, 0.04861401124347165, 0.05156741980186195, 0.058903136346862714, 0.0664161877724768, 0.07964509423130976, 0.06225993223874051, 0.06516507948359951, 0.05572858589031313, 0.057150604188833595, 0.05193790256039319, 0.04873807988619675, 0.048951199336310355, 0.05446642552837986, 0.0660260159114966, 0.06930614370994521, 0.08878037784334655, 0.09183213267270977, 0.12371719360124661, 0.1337088115691775, 0.08870051975719542, 0.0820202126320341, 0.06319697932604489, 0.06336337242092921, 0.06308123016145804, 0.06001908614966648, 0.0728716465392595]
# bu2 = [0.13075023251123985, 0.08370029490183482, 0.07659889183902703, 0.06005314072668309, 0.0612617483623996, 0.05759829290384953, 0.06830263910152948, 0.07449016162365346, 0.059652618017690784, 0.06252119012398691, 0.05706330883183979, 0.0703390395450031, 0.07226471447590939, 0.07109278087610556, 0.05771338491847972, 0.049761061610517264, 0.05803084847545687, 0.05964893125962756, 0.0522865303581627, 0.049551242976812286, 0.0566295800160695, 0.0544770399567651, 0.04789239602923888, 0.05231477349732721, 0.06166885753111661, 0.05987263043424448, 0.06182882627354612, 0.08160824011048763, 0.10025402253622484, 0.08978841527101014, 0.0878657398705505, 0.06897329075665067, 0.0675464789389507, 0.07560344830260082, 0.09347326708386888, 0.08003186135788279, 0.07197845137154149, 0.05395788259413172, 0.05004206364583344, 0.058951583504698306, 0.06566283834307485, 0.054780587079658494, 0.07117641985197318, 0.06055045128126401, 0.08462018106995331, 0.07739999760853272, 0.08041886859573678, 0.07019169184997759, 0.08213307146022113, 0.06797047681887966, 0.08680371483613321, 0.07981573774540368, 0.07339367157805408, 0.06630452773196537, 0.06857059398017554, 0.05338983762093759, 0.05049500410821088, 0.0473787510840294, 0.04581295145312264, 0.045884213172281585, 0.04677283164427253, 0.0513892290318333, 0.05826700886201622, 0.07386528257795008, 0.06799263118612825, 0.06844908331937731, 0.05644314810194286, 0.05420051383027668, 0.06415099677628285, 0.071581655577652, 0.08149963048432872, 0.07763037147297994, 0.05916781392336157, 0.05977437720304502, 0.059488071935819134, 0.05061696844300024, 0.04525486415198735, 0.045079217253317386, 0.04785170459397051, 0.049939424035781325, 0.05646770961507772, 0.06522073946800752, 0.07683915910029078, 0.06065479586175461, 0.06353214375226843, 0.0549411279144302, 0.054676313701300175, 0.050616352033395914, 0.04734584507924835, 0.04784772491429982, 0.05267720899635567, 0.06457662027962999, 0.06758475015334668, 0.08636321759442056, 0.09015612460732722, 0.12087547254591284, 0.1305753369210279, 0.08754054777930052, 0.0805752171711567, 0.06142621503329875, 0.06148742021988337, 0.06098078403080763, 0.05881912155780195, 0.07043518443893777]
# bug = 5.224
# bu2g = 5.535
# flu35 = [0.12051415297410678, 0.11275420083996798, 0.10794759661854048, 0.10919032155796896, 0.09004713966656186, 0.09954853102922931, 0.08421515858228266, 0.09398626165585923, 0.10638680766236594, 0.12409807778955016, 0.16744441730166637, 0.14434403072878843, 0.11577336386781106, 0.1171276428312593, 0.09848820094192672, 0.07592901512628221, 0.07638196438507389, 0.08104729686588949, 0.07877628326918655, 0.08735169827805148, 0.08800405466282543, 0.09116172215629384, 0.0963437285537336, 0.09443245464265702, 0.104379230423865, 0.07943975607350112, 0.12097740156394696, 0.09853114933404322, 0.0939754520481026, 0.11096253271910667, 0.08623840742507145, 0.09446408770845745, 0.08901735645135848, 0.11813610400486617, 0.10317856719761447, 0.1125758295610763, 0.12187150035615693, 0.15414019029147472, 0.12454337592014877, 0.11514155629863206, 0.11002361213353601, 0.10203745642316604, 0.12483133463560556, 0.09312511302805801, 0.11705571145104618, 0.15717062992570877, 0.10591745919892019, 0.10900183172355439, 0.1069973884762338, 0.10262224083704533, 0.0959131853366981, 0.08932256875998047, 0.11085298478630652, 0.10926520012132075, 0.09031229681287795, 0.09484369515255142, 0.08069153402543403, 0.08527387994227482, 0.074256678313623, 0.09754929133246476, 0.09190004076541583, 0.10152154293806326, 0.10198389728635728, 0.08992598939381187, 0.10341836784762107, 0.12020095764105954, 0.09665794591649658, 0.10202530100768228, 0.11967384258622892, 0.09275014754924872, 0.11139085616944952, 0.10627781356961141, 0.10631225487949875, 0.10562318639249624, 0.09621756002639588, 0.1117328355132816, 0.08913653454750217, 0.09261859236466412, 0.07128354692596386, 0.10077550009920051, 0.11065033276095118, 0.0801958724865033, 0.10455285802271642, 0.1021863217679134, 0.09697684980085063, 0.09615051401556836, 0.09791773096146025, 0.10618766819397314, 0.07724739831826967, 0.08449862480458824, 0.08632411101372185, 0.07629010949316078, 0.09772068795488237, 0.07320940943071308, 0.06516069747637028, 0.07138457206049542, 0.07829616313688269, 0.07278208506258485, 0.07192655086821918, 0.08253368188265778, 0.08379243496194479, 0.07108038618078517, 0.08288986369989233, 0.08652804979849481, 0.0908451892949029, 0.09708528769676598, 0.09176117130876262, 0.09968028382372945, 0.11805084999673206, 0.08724892529892553, 0.10293947745797492, 0.09680474327312504, 0.10495590767217149, 0.09973873042857534, 0.09436732467890198, 0.09413811608731243, 0.10880961872168199, 0.1000375086609255, 0.12978554243235987, 0.13126264022456088, 0.14098705098394948, 0.09759912267553282, 0.10919220136218531, 0.15063661947476284, 0.15207915662324575, 0.10102469366235763, 0.1016969585815852, 0.11218173791487844, 0.10334641697489673, 0.1019237625192689, 0.10413151405872119, 0.1198731309495898, 0.12872187615366537, 0.11001475131659125, 0.10931472067609631, 0.0863612246398791, 0.12546808611835986, 0.14597158941132798, 0.16970065050477742, 0.15791545440937776, 0.11799483204049019, 0.1293641257983739, 0.10762660679254997, 0.08229720320175757, 0.09251126484459282, 0.08876964823110556, 0.09062037794048512, 0.08511419869723356, 0.09431501496475322, 0.09664754112903157, 0.10262987864298048, 0.10534839681964721, 0.12535045849010362, 0.13229538855510686, 0.149751556839681, 0.1527233093037911, 0.1449510155003588, 0.1529840371590783, 0.13392637493672777, 0.102420049208216, 0.09507745094485812, 0.11849454716242724, 0.09076429249262939, 0.11646633519630652, 0.10298521127417494, 0.11979654592461607, 0.09542486079333293, 0.14185516786040922, 0.13676402988157452, 0.12140955509318663, 0.10444771256031307, 0.1051950539921123, 0.0748326758831969, 0.08631851724228709, 0.07667507826832239, 0.07762832333688574, 0.08558316742853218, 0.07814040934732311, 0.07187355826808992, 0.09663601467075497, 0.08229160036458022, 0.08786649432037603, 0.08527542100681712, 0.11291941881018513, 0.11700246865291344, 0.1300066646359938, 0.10154138351181957, 0.09118615378946403, 0.1315195971910871, 0.13779970923720847, 0.09791694571003125, 0.10987113075505642, 0.12225278107536927, 0.10270477575008484, 0.10965810067584639, 0.08997604848376324, 0.10397489017146555, 0.10568404276254142, 0.09348601511126965, 0.09604345059326992, 0.1004434791235223, 0.08744390643794972, 0.08574726229245173, 0.09231998780762166, 0.07368706177924716, 0.06924115216105312, 0.0728668903387167, 0.0722951450707935, 0.08996757566644685, 0.06955773818748572, 0.06868764985783633, 0.09120219402349361, 0.09396725573773185, 0.09854811731087347, 0.09530181953122412, 0.09669132166120888, 0.10697996257758655, 0.08659703775344081, 0.10737954110096785, 0.09549840921695782, 0.1067207347203065, 0.10875678319311628, 0.07608821059188972, 0.1023897602133863, 0.10980423964207736, 0.09247705655418507, 0.09062434139605387, 0.06402457835067726, 0.06811434501351038, 0.08222944477883216, 0.07305733948709066, 0.09279204887120089, 0.08895244695117523, 0.07735913480689263, 0.08932000701122367, 0.1061684149288319, 0.09351319150357258, 0.08958970055409407, 0.09658364455947138, 0.08654214113353582, 0.08663514074571212, 0.09163262942738987, 0.08706697150594792, 0.09223699702644389, 0.0884348651978905, 0.10288484903934754, 0.08003934572971193, 0.08750522500626699, 0.08393625682616807, 0.09250495678952222, 0.0867414451744317, 0.093723806471141, 0.08437348909377593, 0.07567234508935082, 0.09386459559726949, 0.0969202619907348, 0.09757371013085589, 0.09264030965379218, 0.09324102371702638, 0.08948791846429523, 0.09253436479877333, 0.09685732346333446, 0.11427644632963188, 0.10047363344147375, 0.08169545713383328, 0.08588938978649859, 0.08581961803013557, 0.10592372945057336, 0.11906134461319628, 0.11300375832572856, 0.1284386293041953, 0.12357133271187472, 0.14409410385884328, 0.13572005128716141, 0.12391968775355088, 0.14267732765841531, 0.10757374730914299, 0.09717435573974169, 0.10592912787520774, 0.10526154861159912, 0.09609900952605592, 0.11258372609775999, 0.12173099624944382, 0.10050356767787835, 0.10409871693315015, 0.11576701593307397, 0.1187134685883648, 0.13348619422055238, 0.11020986963304846, 0.09947659144089437, 0.08696160624525122, 0.08095633954992758, 0.08248514603175036, 0.09978205807527843, 0.08447246857672716, 0.10492721094255077, 0.07763025948502011, 0.09002697548782887, 0.08237878443544969, 0.10442513325125227, 0.08206530749989774, 0.09532293309312834, 0.0824314867482884, 0.09409276695091778, 0.08436911473204306, 0.09694999686145071, 0.08756237866870145, 0.08203659997747793, 0.09181530106359702, 0.09826661815960297, 0.08969155952339719, 0.09936630799634791, 0.09689717699992645, 0.08168005905249254, 0.08587359639684922, 0.07272879682333333, 0.09383070906666789, 0.07945155284630416, 0.07913844365117224, 0.09908764517038898, 0.10672706820961814, 0.1070882198509801, 0.12562758292460918, 0.08784471735396657, 0.07591912215689066, 0.06834385374727033, 0.07480174494706922, 0.0753382466855293, 0.08567731207306704, 0.08843281727084493, 0.07371699705721395, 0.07819202432277025, 0.09410036132688925, 0.08319220956244433, 0.11368509485447471, 0.10119350543279385, 0.11273345504310411, 0.12506838771264306, 0.1314510352289631, 0.11743393747073522, 0.14644286646227178, 0.12289334063290902, 0.10981336143525502, 0.07637036095483804, 0.09353475171048727, 0.09442129472935207, 0.10247875132388262, 0.12795469019904931, 0.12047113635947397, 0.1265052035864919, 0.14706011333547894, 0.17949479319298522, 0.16388656978704344, 0.1602911154629947, 0.15897563934898137, 0.1269287546528816, 0.14186150247382234, 0.10185993645131089, 0.0971197333254693, 0.09219596977270698, 0.09960617920196414, 0.08765290336065876, 0.07765043219391546, 0.08831291300628646, 0.08525029192921527, 0.07490025492876379, 0.10297438288313487, 0.10557761370940047, 0.08275853770636687, 0.07819104052864573, 0.0963296915685366, 0.09543405715412148, 0.0741836528231095, 0.08108598113514624, 0.0819263339072641, 0.09028808670783126, 0.07809514156188034, 0.07436919570389325, 0.09407806121899226, 0.09530694522295835, 0.09795993491008856, 0.08054024519895887, 0.07645778029075762, 0.09281617822745332, 0.10505140830365205, 0.08496062697900336, 0.10205941912528387, 0.09176784966303307, 0.0661806435703381, 0.07731574739888564, 0.07728349945973616, 0.07629847539938256, 0.05824809789957406, 0.0761207433243294, 0.08721464407904003, 0.08282474458525665, 0.09222901893285111, 0.0726821846496986, 0.09023028606466377, 0.062354348668192, 0.07016254731824337, 0.07501685116791151, 0.0662922367632453, 0.07147702672562771, 0.09990344955792481, 0.07214268639718359, 0.08149916250308421, 0.07213668644340757, 0.07759354008027655, 0.07108429093030567, 0.08810550035887822, 0.0730221236910951, 0.07118925437314726, 0.0728120753363693, 0.06566306139471847, 0.10987977234889701, 0.08485167767667502, 0.07352942023811213, 0.08860842172396463, 0.0757352680166858, 0.07135076618705702, 0.070992704993627, 0.06904174370475505, 0.07657096900803635, 0.0751249441840459, 0.07940194562189397, 0.07348665229501626, 0.08121682412228644, 0.07953603137121268, 0.0780202803601838, 0.08477228356749593, 0.06824750140706423, 0.07506585563279024, 0.0808262865909196, 0.06696584275721391, 0.07263843303438369, 0.08671098916002451, 0.07732592975522914, 0.08198622633360882, 0.08281371381772432, 0.06484811271182868, 0.07349425189860026, 0.09390537032231366, 0.09072303691388729, 0.10018860950591124, 0.08568590887041795, 0.07740896580486467, 0.09518768975227819, 0.0926934471743576, 0.10093594325565544, 0.13042293273025068, 0.10705960762891116, 0.09572700259760604, 0.09539644910182243, 0.08613037002508185, 0.11519581580495715, 0.07926502011729819, 0.09090448735924518, 0.09231746053585904, 0.09640004773583308, 0.09416607574968135, 0.11505155349499661, 0.11238739942877038, 0.11883294977833865, 0.13434152207840708, 0.13024433151649834, 0.15069583671819148, 0.13325355819403126, 0.12826588789912752, 0.13397288291554318,
#         0.12150241874863578, 0.12188271485996809, 0.1399221363751749, 0.11528480909976442, 0.10872455872365992, 0.08282338326456956, 0.10999146068733216, 0.09977563933721295, 0.09852995343940829, 0.1157304509565664, 0.12328715766534412, 0.1099379295032337, 0.1137495729130895, 0.14955601757079284, 0.12698575400195564, 0.14823086070841607, 0.13798854965483542, 0.11991076204175087, 0.10982432404558208, 0.13678899372312925, 0.143381991821077, 0.1277549302345289, 0.11081036696724811, 0.12210240385048503, 0.16678155218294655, 0.10649834702295931, 0.13877432044471896, 0.1611795513471144, 0.11146014230140124, 0.10647903584400849, 0.09903169635041095, 0.1173291037448693, 0.08559980340182281, 0.07819171727687915, 0.09668622535373281, 0.104169986275608, 0.0985363450869298, 0.1041834678968034, 0.14554540725500562, 0.10626274367288503, 0.1298197165952421, 0.10780385644251568, 0.10046998122557341, 0.0938791843393075, 0.08906148217555525, 0.07292932899783237, 0.0805766187709197, 0.07755468136254098, 0.09011814178464757, 0.07843640762258682, 0.08797543218270625, 0.09179811092908446, 0.08803446509389146, 0.08120203823852241, 0.10735472081885152, 0.10353839229503099, 0.11626285437868905, 0.09699937342238442, 0.10388129797196688, 0.07897098081007378, 0.09575357853953866, 0.1054056197160424, 0.11313517244221888, 0.11469804332171882, 0.1244567820507296, 0.13452871161771626, 0.13335474861891292, 0.1311322121306808, 0.13449722227640537, 0.11600223811537258, 0.14595595478571122, 0.11665927641331876, 0.10208769680493499, 0.12326124999685588, 0.12154169137684691, 0.10755225306088675, 0.09083213038023663, 0.09949810885804426, 0.10671998832415396, 0.08275921175729357, 0.10181677244910062, 0.10647541257188076, 0.09038044860953212, 0.07545335656699728, 0.0749327789385896, 0.09003096038334142, 0.07759015271441277, 0.08705505598590747, 0.09556091167813173, 0.08462270330400623, 0.11536734971310414, 0.09597509699279404, 0.109777383837068, 0.10080706882629957, 0.08553563952571434, 0.08976170453490305, 0.09633504976886847, 0.09952450634289148, 0.10085457567014729, 0.111981686445848, 0.09453205694585032, 0.1261868846652965, 0.11173962735237633, 0.1000673579135482, 0.08626317781132674, 0.09899394646952098, 0.08513965894936602, 0.1020677612284503, 0.10081653850206407, 0.09243060317766391, 0.10952693004078067, 0.1183522451278338, 0.09541003521813612, 0.0860099422488852, 0.08933020296321889, 0.09307979778978036, 0.07743345122468497, 0.08570244762372783, 0.09448992647425015, 0.11189818191447082, 0.08967335113154494, 0.07281851394827889, 0.07005898526207838, 0.05789314033130137, 0.07117017267169724, 0.06305258418036692, 0.07610255869670951, 0.07710634612169982, 0.07439906103062031, 0.07878723273978006, 0.08459435174531259, 0.09707611721063952, 0.08537262321959278, 0.08454735107259802, 0.08954729422953826, 0.09894764031795496, 0.09833912169360576, 0.09331095902376592, 0.1068849295945644, 0.09390093923663156, 0.09905528602854864, 0.0903056330494839, 0.10324021123513835, 0.10049094932376801, 0.11623201268930831, 0.1087287911182559, 0.10680976866352151, 0.1081226474272503, 0.10665710211296905, 0.11378340098287884, 0.10312297023872083, 0.1259170442023604, 0.13474483283448843, 0.10050984154225441, 0.12608432177139997, 0.0870841292227409, 0.115383057835415, 0.10800615127325766, 0.09554277928093283, 0.0939887830244191, 0.10473137533164502, 0.10668683678305492, 0.10104616163289709, 0.0986632225169941, 0.10680400223870903, 0.12228668486263705, 0.16024635428609765, 0.16052984626698105, 0.12260974552938038, 0.10314288357116848, 0.09956237787018823, 0.10306077587243127, 0.09263526090514435, 0.09120037154211341, 0.08167510361298976, 0.09601527217882073, 0.08621846740121393, 0.11595318843671423, 0.08542479110165654, 0.10046994294173957, 0.11346521010950887, 0.14563849011952382, 0.1811719470216472, 0.167181334213608, 0.12848554617235064, 0.10963728414266373, 0.13697669333292029, 0.10559070199233621, 0.10218582364123542, 0.08607636300935946, 0.09014462549820239, 0.09410645271609919, 0.10197033831427026, 0.09414490375779432, 0.11035699956624888, 0.10157667503612598, 0.10220372595211813, 0.15885208976514106, 0.1159435740538872, 0.10013313828047074, 0.09989999180795224, 0.07723803094537499, 0.09028364850410367, 0.08567790728402168, 0.06600794838404288, 0.06762638454934056, 0.08996901042020179, 0.08599432304510271, 0.06923904166618164, 0.10173906596796396, 0.10482199928298713, 0.11402480313971829, 0.08603444873006474, 0.11483053588090497, 0.1181407941727605, 0.1329637034135663, 0.127401474115475, 0.12875735280277653, 0.10435952897078055, 0.0966939060474279, 0.10180137372515476, 0.11131128876751416, 0.10427342258956815, 0.1112497449064264, 0.08596553490815531, 0.07705469708593997, 0.09902238443191716, 0.10262544677952835, 0.07128808317675399, 0.08702712815163205, 0.08175109760102202, 0.09573846815978113, 0.0827274670036626, 0.07977440319579059, 0.09039114368982126, 0.06514464187248663, 0.09936299504832333, 0.07326527542589228, 0.07239335221702775, 0.07357352990414603, 0.0861811743679692, 0.1081981554528051, 0.0970899961460331, 0.10008326698638996, 0.10263311701422578, 0.0909506962808882, 0.10733259737017461, 0.08986595214272247, 0.11180254993745117, 0.0990715002461433, 0.09103140141550768, 0.09497386074222118, 0.10909063810225345, 0.09611964203263842, 0.07965198180534744, 0.06923701079575191, 0.09359953880670797, 0.061431954884589246, 0.06850663494514157, 0.07820217623001942, 0.08373876443578047, 0.07854639316551114, 0.09044945171432937, 0.10450442224948699, 0.09566027571139477, 0.10607288810513824, 0.09311611709635034, 0.10640243329088457, 0.08829443501697269, 0.0947350707720777, 0.08952444206109653, 0.10172080604831121, 0.10573966563795617, 0.08646140891242127, 0.09338485424567944, 0.07554339919706247, 0.09284831992822291, 0.09307868809516189, 0.07454125999091651, 0.08521353773319237, 0.08478126371956617, 0.09005862286722327, 0.0982923214179207, 0.11217754030338707, 0.09960301714016488, 0.12691652708631784, 0.0951111006632452, 0.07726090460875423, 0.09099298077615142, 0.08968892184942402, 0.07789000101595771, 0.08304045707854744, 0.09559804351880913, 0.07762759063118578, 0.09809092625779992, 0.08914620621593261, 0.11405061339158813, 0.11280108740862352, 0.13471899289483502, 0.10987938668096654, 0.12546645684226282, 0.13361126856335978, 0.13543859579641648, 0.12227789393922386, 0.10101260222334728, 0.11760978326018973, 0.11483688679436203, 0.09826468978029278, 0.09654267565288284, 0.07840822346900501, 0.09321756598251406, 0.119341013166947, 0.10750570078060097, 0.10866402585832682, 0.11169725880146736, 0.10447178149216724, 0.09664843251299696, 0.11295130233818614, 0.08283005122034558, 0.07612074747759225, 0.0967551406961484, 0.09174965295357576, 0.10200190049757438, 0.10303506322098367, 0.08194925277532308, 0.08931468871516365, 0.09693476217279985, 0.10211575287343311, 0.09943331352578315, 0.09843221448464014, 0.08820320608950422, 0.09303264772121402, 0.0854191983869728, 0.09016140635040579, 0.08451396389759823, 0.08977589403238921, 0.0938603540336962, 0.10469913094086902, 0.11419482439691013, 0.08862632537392909, 0.08143456684074248, 0.09449810187852205, 0.09113792771510819, 0.07894409344061833, 0.08478923655075614, 0.07766050241745995, 0.10408387312384983, 0.09276781919443985, 0.09522558861182154, 0.10605563424283455, 0.12498527529197981, 0.08094213105026068, 0.06841622803763085, 0.07467929303063335, 0.07655213923178592, 0.07809411895398986, 0.08829480892345902, 0.09276428857960314, 0.0680089005327532, 0.07164437555982642, 0.09100105070550311, 0.09321477735432311, 0.1117821089221876, 0.09915444155788343, 0.09978445316888897, 0.11518668222961354, 0.10908805604163498, 0.10131517863412619, 0.13302596650908968, 0.11518600688659099, 0.09292439455136967, 0.08471519311120644, 0.09101270681105841, 0.10465967879525584, 0.11541225654471562, 0.11516378770751794, 0.11976470451467681, 0.14614668807972836, 0.1331870237763528, 0.16899456189260637, 0.1937072793424529, 0.1744839739316562, 0.13780268936903262, 0.135227528638262, 0.10374257565596011, 0.11415539295803262, 0.1149198501760039, 0.10422452649044879, 0.09797948828183035, 0.0933459472987384, 0.09452795160367473, 0.07824021953161822, 0.08593083198639598, 0.0900149829217592, 0.09372241499001652, 0.09313554985365503, 0.08525587982649117, 0.07548028706748817, 0.08040182495768407, 0.09077024166787867, 0.08782889758296196, 0.09545575202884693, 0.08251228770266496, 0.08486536667498328, 0.07831437878665398, 0.07888806113096292, 0.08310974068358253, 0.10491139964306105, 0.08911281163465147, 0.07062465209557175, 0.08257270070798767, 0.09221278629339301, 0.08728298083192157, 0.08477187784641244, 0.08945263237301124, 0.08161540153732827, 0.08818748803425495, 0.09586648335165232, 0.09399495267813135, 0.07636275435030522, 0.07938040035239834, 0.06353744616341239, 0.07507803304056919, 0.07103580813889626, 0.08273847124424386, 0.06923301425803444, 0.08437762957245253, 0.07573303378888707, 0.09178670238050407, 0.09467489523700423, 0.07614942673758132, 0.09400690092488496, 0.0797597468143625, 0.08933257104489498, 0.07382028313917566, 0.06742092810052866, 0.08927577034269826, 0.08331397894881057, 0.06759053285807794, 0.07392802448564796, 0.08980759744271968, 0.06347888238663697, 0.06282453067894309, 0.08170411864106365, 0.08724415464948229, 0.07550271489548693, 0.07509005878447925, 0.07358891393850488, 0.07503181437088721, 0.07746125015591848, 0.08591778609950318, 0.08454418679656796, 0.05885688865604854, 0.08311471052747157, 0.07393860672782968, 0.08608488805312854, 0.07274602199149899, 0.0731998068149851, 0.06813653750223372, 0.06011543507229756, 0.06956691391673446, 0.06613660309514313, 0.09037233988295755, 0.07151206054220016, 0.078429650116967, 0.08792441020690422, 0.07419705342594907, 0.07087064363955452, 0.07664227541750813, 0.09544507445308918,
#         0.07772244294518266, 0.09022095238211131, 0.10197132780139971, 0.09377764055731487, 0.090372245955354, 0.08421531971416761, 0.08666808627200949, 0.08886915880278261, 0.1128509422788857, 0.10549083951374175, 0.1081086041794442, 0.09785794433192607, 0.09459861622309668, 0.10054590382204197, 0.07670457047175556, 0.07173728095651453, 0.09284096550699356, 0.10263107156346951, 0.09047278029756636, 0.1055619704596191, 0.10812653777039243, 0.12413782263939627, 0.14135978544366115, 0.13470200765017015, 0.13098935903419578, 0.15260307410044213, 0.15548484849256458, 0.17292202967480244, 0.12245475341132712, 0.11371021263595024, 0.10667990465642255, 0.12735645841423904, 0.11514420904472097, 0.11497155820122906, 0.09898410661245396, 0.10976955702895116, 0.10782979111795964, 0.1155123418710545, 0.12196805563059346, 0.13231182842862127, 0.13795848417522363, 0.1549766641993859, 0.13077135639522366, 0.12117073159734532, 0.12499767207844711, 0.15931495023097747, 0.1242810066760143, 0.10597101690961289, 0.1474200930973018, 0.1391418737064337, 0.13281488222453294, 0.14081563500536645, 0.10640038848182849, 0.10749992782187046, 0.09382273096964967, 0.09528326848851687, 0.09576090176711614, 0.06449005831850951, 0.1021567645074019, 0.09564347036749281, 0.13027980797708866, 0.11541026489805034, 0.13793947750509214, 0.11887096515328129, 0.12531040423654927, 0.09767293280447792, 0.11102071952769772, 0.09857683682619219, 0.08870593796017473, 0.07882432720158669, 0.07071499223025104, 0.07865042185123049, 0.0759973524676813, 0.09507809141798929, 0.10901473376463215, 0.0973199371980373, 0.11702303030977772, 0.09172768664843187, 0.09580746543214916, 0.0935222430137022, 0.09790719866040766, 0.08233261870547409, 0.12390645531890085, 0.10074014490129124, 0.09043886430713197, 0.10962170888109413, 0.10137485897968398, 0.1206127011866065, 0.11782865278660347, 0.15547106768292035, 0.1258346678605102, 0.13768010309116724, 0.10849648211111657, 0.1026103632369449, 0.10262541284679379, 0.11931892559886019, 0.11688776273262652, 0.11448103515824319, 0.12858197338826086, 0.119686250112045, 0.09758801083118764, 0.09452000036853078, 0.1007761825043152, 0.11503545100971943, 0.12975213304040373, 0.1507950972260763, 0.09348248631439321, 0.10651014755265766, 0.07865290863582194, 0.08656108981610651, 0.08007736217259677, 0.08240338559378, 0.09840950435717792, 0.0867773202336392, 0.09506857029818885, 0.08829026238752531, 0.11238518878370957, 0.13348279788138237, 0.11172869128489382, 0.10690763571374806, 0.10282779850886742, 0.11044964879092448, 0.09457322152987226, 0.10869713347535279, 0.09584896090900435, 0.10263548780076694, 0.11231057595416967, 0.09080676116897837, 0.0827180044900009, 0.08650425152943958, 0.08067120440244639, 0.09775421696793798, 0.09543874307849474, 0.09295488933136963, 0.09914426064212171, 0.09671837997603415, 0.09982485652880708, 0.07838453793692114, 0.08782667011316221, 0.09192431961434183, 0.08589493953880094, 0.10435499223336862, 0.09789127380789617, 0.09871795403596427, 0.08078181024063918, 0.07148782943194587, 0.0698782015631532, 0.06773657290517858, 0.07688487403474316, 0.07301436065853131, 0.0694560795034922, 0.06978479276190921, 0.0633686879151672, 0.07340270215648127, 0.0813098255688119, 0.08054922689942637, 0.07828892508405383, 0.09229164478307629, 0.08555048319687304, 0.08271668745807716, 0.08909102984942992, 0.07543282868937254, 0.08759696830173948, 0.08898662700527751, 0.1008669953450354, 0.09848353448125026, 0.08189515196133684, 0.10031631551817399, 0.092157308641399, 0.11849381368460342, 0.10871248138951924, 0.12120712190712143, 0.12205724646343122, 0.11829966729280914, 0.1257819713660891, 0.13213866811462122, 0.13357276991347047, 0.11379652365469685, 0.1050268491683051, 0.11560455090175467, 0.14924563859887405, 0.11451954507907418, 0.11387776439086872, 0.12016575153236869, 0.0988986307880539, 0.10350724393069144, 0.1077847940478338, 0.10203547976399491, 0.10819248349848828, 0.13607841302207865, 0.17174175457082735, 0.1701451934484339, 0.14929895308759097, 0.1202493828275792, 0.09624588180361549, 0.09806604935840293, 0.09817645317997649, 0.08657804144437115, 0.0879964067392299, 0.09341634000503081, 0.08953587869896405, 0.0823485566682921, 0.0930729586050481, 0.10686211569937347, 0.09870867097748527, 0.1171816793024132, 0.15488430872152587, 0.14458921760462262, 0.1517507848667852, 0.11678626343767332, 0.14534664587732804, 0.1199421600293366, 0.11247675069517898, 0.0956523656673501, 0.09857602785619314, 0.0994076555911408, 0.10414399764424459, 0.09771456434885602, 0.1323273788416557, 0.12923865890619735, 0.11237326452755542, 0.1364906352658923, 0.12138090751855869, 0.09120628575823396, 0.08228378021690481, 0.08949611868578798, 0.08262468786113578, 0.09143235903167993, 0.099520527115357, 0.08199400070425321, 0.07657046130113467, 0.09016085626401267, 0.08686068891385208, 0.08770778696024767, 0.09037106227115088, 0.12580298155370365, 0.0984376315012987, 0.11968637205065887, 0.0924686814901383, 0.0965865107148766, 0.10695358311222485, 0.12832220930092963, 0.10643883580472205, 0.0965697311214576, 0.09600259332322354, 0.09542180419334424, 0.11293773503042082, 0.09779481849303992, 0.08992834520730308, 0.08486794118774794, 0.07335040561123549, 0.08212022539001584, 0.09577535181633921, 0.07399452639447254, 0.1015567789976943, 0.08444382455685191, 0.07357803572524797, 0.0737367600384784, 0.08139526537120055, 0.08725357611432473, 0.08204344818228314, 0.07443666136438103, 0.07790390110635935, 0.10906439605427777, 0.08311579663476602, 0.09286365521815508, 0.0925816135045776, 0.11110341796241606, 0.10712064756690286, 0.1050811079520895, 0.0977451509155226, 0.10462507328731568, 0.12892488236783786, 0.12141995234657023, 0.11657933634132217, 0.09874762518949175, 0.08719718645610897, 0.09462527133368584, 0.09378678591279921, 0.07220626517311383, 0.06686882348997139, 0.06273288072635766, 0.0848631846828119, 0.08858879393732645, 0.08784391071477553, 0.09139379011381973, 0.09674576834474638, 0.11483784672547179, 0.12008956994160223, 0.08824886838057353, 0.09817323534425806, 0.09056843983112663, 0.08684976074956145, 0.10019504225479459, 0.09331134244432894, 0.09636005663688232, 0.0980724850194507, 0.08758940492008904, 0.08381069775352577, 0.08898561333018945, 0.09270188035413458, 0.09044589254583926, 0.0985466784501283, 0.09107520830977898, 0.1029215247698518, 0.09206876661278261, 0.091902470256292, 0.10423980031922565, 0.10943347094511402, 0.11917061881490272, 0.09860429007543708, 0.09837118245418967, 0.0913827946969974, 0.09532071642031247, 0.1032806294849311, 0.10251855989410526, 0.0878672311314752, 0.09055810884378214, 0.0878625816033754, 0.11796292976791353, 0.13204382112579702, 0.11218179457617047, 0.127151981398926, 0.11750787398206688, 0.16375213072330913, 0.11715551723562444, 0.09730888659902363, 0.14330508062941974, 0.11046248440893627, 0.12038423364704265, 0.09806137561143917, 0.10551506430096956, 0.1117776978724461, 0.1099534806078084, 0.10462345959223267, 0.09660956165024132, 0.11086355866255258, 0.1089444681195373, 0.11670819197778941, 0.09684280635161005, 0.10634021008377015, 0.10292813734063827, 0.0864682724897134, 0.06997798579763194, 0.08677428132515169, 0.09496595922313372, 0.11841567392450679, 0.10116387210765633, 0.0787501730302218, 0.09288935816023401, 0.08812299354034199, 0.07871954829448664, 0.08450851976734702, 0.09492901947721287, 0.0729297483219145, 0.11038597133132957, 0.09399897392793795, 0.08852548652351136, 0.07986138061802758, 0.0834235446863165, 0.08033916253932359, 0.0974324833504921, 0.10468050986621656, 0.08600120159016954, 0.1031968313497405, 0.08554448405671644, 0.1053005760382483, 0.09103196323076576, 0.10509400958177935, 0.0936911472018029, 0.10087778370526991, 0.10539452262630442, 0.09528274444312039, 0.1110419827344535, 0.11690789644772386, 0.06761435704992715, 0.07605552484038257, 0.07083436270981681, 0.07079945828529641, 0.07227752401774086, 0.08898833709541426, 0.11147708907448428, 0.0814017019857891, 0.07329958790685127, 0.08649475041702936, 0.08519073301372339, 0.11024109265404891, 0.1018847675535292, 0.11672556190677216, 0.13470093501814864, 0.14003535317564633, 0.13533932378772523, 0.13321145617201333, 0.13075299393723464, 0.10989711948587777, 0.10283103791252166, 0.0991988403362065, 0.10634548390363577, 0.10179168675318907, 0.12325431274364992, 0.1344580925918354, 0.12666383578403026, 0.14836874922981716, 0.16997985176702787, 0.17422560437244322, 0.15966309139417004, 0.13571542016039317, 0.13783523998232972, 0.14090922560479338, 0.11468192752576307, 0.10816358130678407, 0.10351754914247406, 0.1086213638577595, 0.10292495784479228, 0.10133101473564224, 0.09046676780077753, 0.10537507791699352, 0.08762607949093935, 0.07162189779080906, 0.09043277408468434, 0.09774523886704468, 0.08630792641244629, 0.0903256645014143, 0.09309686715546307, 0.08327506184872277, 0.07708389208168033, 0.09332900129545758, 0.08546937782797828, 0.06839823419022817, 0.08301907487555601, 0.08320397885808892, 0.08762827238635335, 0.0789549195252456, 0.08125455310016712, 0.09298581453194732, 0.07494576059012875, 0.09104543105418818, 0.08751177596330673, 0.0793381578689699, 0.0977465977000191, 0.07052260260656676, 0.09884172613371331, 0.07377159263014646, 0.07971926052343103, 0.0646401530762152, 0.07330194401116151, 0.07112916654704349, 0.0843597413529039, 0.07980660184130971, 0.06573113409991602, 0.08099523920426754, 0.0797952456424722, 0.07802078970551574, 0.07186668778601243, 0.06294431481028326, 0.06399849018964993, 0.07457657005406844, 0.09251542208146465, 0.06887738340818686, 0.06852884866048348, 0.0714859630775235, 0.07514678851557781, 0.05719580667675557, 0.0679578670992752, 0.08288555774569041, 0.07642477141535783, 0.08080110961748772, 0.0831092097225196, 0.08290151718834705, 0.0819717161497152, 0.07853915451313544,
#         0.07213491958194584, 0.07706200244969649, 0.07731326674767161, 0.09162222226328375, 0.07321062349684543, 0.07608394792043845, 0.07633211260989811, 0.07464125239361871, 0.07044693165014827, 0.06904746473382674, 0.08378401636073561, 0.0954403505143378, 0.06424764291546207, 0.06913273900733373, 0.07649931828799543, 0.06361479907262023, 0.06452425603233436, 0.08199680599585567, 0.09568185113940393, 0.07109014285933622, 0.061428082360488544, 0.08473728207691086, 0.08626014724726444, 0.07817651999813059, 0.07874944011645371, 0.10445542057156103, 0.08421615681804448, 0.09550511662831823, 0.10279866915135544, 0.09482790309370757, 0.07819408581297023, 0.09999915980884763, 0.11187617558920182, 0.0945283986436285, 0.10673070754244383, 0.07970844636701237, 0.11486610703060932, 0.076519715870793, 0.0965939928895104, 0.08485819274359102, 0.08828541853427388, 0.08669500850893845, 0.11375273894795666, 0.13226296598938975, 0.12498264477811627, 0.12630702486353973, 0.12572087601559873, 0.13561152941195506, 0.13155147688862992, 0.14452573710934627, 0.15100104952563595, 0.13106253497404835, 0.10590766359153876, 0.09663765717732409, 0.121628947455545, 0.10627535152054696, 0.11162219629265564, 0.11810968387811419, 0.10118300189925793, 0.09651732409320657, 0.10629420736728393, 0.11940133538459378, 0.1327934710685869, 0.12757943154313242, 0.11825991707206497, 0.12431478215440676, 0.11003764815244758, 0.14669991061669393, 0.10343302189975091, 0.11892579249265918, 0.15475296420442042, 0.13893290606381622, 0.13928429804005182, 0.122155105315967, 0.14771549225368716]

# flu25 = [0.13061189761118053, 0.13119536960171643, 0.14993958140957744, 0.14082956117953802, 0.11342526122752758, 0.1193898810770981, 0.09836657861130842, 0.12269153237336398, 0.12927311152628015, 0.16037475941067292, 0.17172940445635568, 0.16788013651849762, 0.1615150661831958, 0.13059970424076223, 0.10972447215166735, 0.12548651123979798, 0.09810578890440079, 0.1014631670540831, 0.08547778845250155, 0.10706839082520078, 0.09021724765957344, 0.10437681384577656, 0.11041218891724747, 0.10820156338426432, 0.10872550614844048, 0.1181408783201439, 0.11464701008176179, 0.12312389966491062, 0.09594612311488869, 0.12296696664741057, 0.11976486339234965, 0.0887094578683654, 0.12377448031812571, 0.11319004758455, 0.12322351165895744, 0.15089277917292968, 0.15011176161109946, 0.15672095405486697, 0.16058800209224178, 0.1501830402300431, 0.11162332912047848, 0.13550342173935795, 0.12383382484781831, 0.12833310945923976, 0.12874682618036853, 0.14844235253711038, 0.13203750955712648, 0.13020723762540848, 0.11186725040512277, 0.11517134521582996, 0.1290192607956158, 0.1552898598162453, 0.12860431896458085, 0.19007159338141272, 0.13088597487395898, 0.12466655466080344, 0.12527284800632063, 0.1156617159612832, 0.09189276804709236, 0.1367533990697712, 0.08702109512103173, 0.11814172167369999, 0.10186290102266307, 0.10460120895930061, 0.14036700298578605, 0.11530633620055054, 0.15979049575327287, 0.1188599165091188, 0.10850630155274268, 0.1270884439252759, 0.11869069745646192, 0.14035161960208145, 0.14230515077519026, 0.11784280495952107, 0.13307802174033778, 0.11429689517969419, 0.1273209682571391, 0.13045566074017256, 0.10657707790730302, 0.1162779843676147, 0.1270900922233946, 0.14372229771388845, 0.13117551553671664, 0.15914741021401488, 0.16483458878914825, 0.13340597966172146, 0.116670942029962, 0.10900462449334705, 0.08413119024276414, 0.10885296450623295, 0.1067815804636425, 0.10364193763176897, 0.0972398598222318, 0.09856021404471477, 0.0864028141826022, 0.08918528946923057, 0.09334076400505073, 0.10161625578162048, 0.08586374663381444, 0.09998795714661986, 0.11881971779618677, 0.09436873435475186, 0.10638262356680445, 0.11517268048135094, 0.10670898910651265, 0.1281727239172119, 0.09755030299238054, 0.09751559151989658, 0.11965910030102056, 0.12446850667665003, 0.09709522753983049, 0.12333936146940945, 0.11646920732188401, 0.12225353007369553, 0.11578074590224206, 0.12348924916472419, 0.14472422814420044, 0.12318973910444528, 0.1022081331732358, 0.14360518335278794, 0.15195804092349785, 0.10312023863072098, 0.1555585805454209, 0.19011260569921667, 0.1631753555576063, 0.1316154059608262, 0.12598648936671775, 0.11810478219377998, 0.13834060343732996, 0.11740788978514588, 0.10675072454877298, 0.127693479996909, 0.10225253922042395, 0.16135866907761828, 0.10985667929367683, 0.09641006835018984, 0.1284222482384637, 0.1595992143710411, 0.182497604579119, 0.21878425367576795, 0.1743005402511405, 0.1276597945501583, 0.11167999635658746, 0.10921484843396856, 0.09713861013393708, 0.10628027580500624, 0.1090188367416498, 0.09337149110161386, 0.11450688478242763, 0.1012369288998789, 0.10668455092138131, 0.11065679563505078, 0.14542460738263785, 0.12233365094743091, 0.18447534607136445, 0.1843467856395532, 0.1592482899491557, 0.1617181318977681, 0.12274409884639124, 0.11972283887075467, 0.09769301052912224, 0.12211576684765983, 0.12936165976195416, 0.1030762948507376, 0.09516318657764335, 0.09823379266636195, 0.1276926777260176, 0.15398924390873978, 0.137919631741748, 0.13615392149110778, 0.12859833816567742, 0.10154330249702075, 0.11607082238323618, 0.10392976305794745, 0.08851266789668155, 0.09066068626488984, 0.09060119174819048, 0.07278027100738467, 0.09988515906581565, 0.10050932449539514, 0.10826210916781473, 0.10369371736880831, 0.1283277877509955, 0.1478022610953153, 0.1275194582882052, 0.1342657673021143, 0.14208960034547083, 0.10987835225279144, 0.16426812276227834, 0.1402247956904285, 0.1376673172386711, 0.13712327848712014, 0.14665723231792677, 0.13152473628402714, 0.12794144578602246, 0.11406299663964427, 0.1253910311760309, 0.1187429169988727, 0.09887186037665809, 0.09756982158580736, 0.10253032623905747, 0.10631060328272057, 0.09377550343980956, 0.11036781847842751, 0.08824592397398003, 0.09583799444114821, 0.10192012598098432, 0.09249876868935575, 0.09482295051464577, 0.08451641600037083, 0.08409105527744075, 0.11508881400141471, 0.12156347552065623, 0.10020179871197228, 0.12715300795136356, 0.10193801910305719, 0.12249405022783304, 0.12719613508243563, 0.11667733129308866, 0.12100879290772573, 0.15032964618000502, 0.1477795115394584, 0.11588259813645647, 0.1002350749052536, 0.10427302592105722, 0.09127078313539208, 0.10927830562361099, 0.08095552161195982, 0.08783152497576162, 0.08123626591750474, 0.07851544708873097, 0.1049363881086034, 0.09844889519050833, 0.11178998990478212, 0.10823480614453221, 0.12581927966683096, 0.1329388424489041, 0.10115958431136295, 0.12026272406325224, 0.1009037988683261, 0.09277134030838467, 0.09354839459854905, 0.12195831173364591, 0.10256891303183527, 0.10013114702226801, 0.11696795919123971, 0.0943238551570997, 0.08242405419417702, 0.09791843695317169, 0.10875380401224943, 0.08966485640167915, 0.11761423792183522, 0.12825222484053952, 0.10446540372944357, 0.09808765369962424, 0.10857450957278163, 0.12550208143008768, 0.12490656779153413, 0.10480812576519141, 0.09651854727847213, 0.10482892271091142, 0.08116911959982519, 0.10422955850300615, 0.11424452692503867, 0.09696260687942097, 0.1509374723573183, 0.10422328352960637, 0.11563090403226252, 0.14834670657875362, 0.13562179032244565, 0.15427883179327473, 0.15963720961876943, 0.1761737312720047, 0.15435601775187252, 0.1522826971704071, 0.16708303593454663, 0.12781443965829978, 0.11517591168994937, 0.12636986525711658, 0.12674885184214982, 0.13715457661913077, 0.11397742524140896, 0.13114385685067098, 0.14112150574903648, 0.11479790935844437, 0.1200453317635747, 0.11753929649737596, 0.12963973051933692, 0.09980617212836401, 0.11483670251202968, 0.10616295443015672, 0.08017638571789568, 0.11739472903175609, 0.0897920273997673, 0.10867148304324042, 0.11393802844665973, 0.11409404625167849, 0.1007300164861614, 0.11668347137982515, 0.12533703135581573, 0.11475647561367844, 0.13481506105826246, 0.11591222900303218, 0.11562332144554227, 0.1079224734746243, 0.10944062951845202, 0.09210920699201904, 0.10673510815349377, 0.10643702561716753, 0.12025499085243983, 0.11105588432619024, 0.09935709393858459, 0.12158534900596006, 0.08861097516450886, 0.10950640905411684, 0.11421213410893637, 0.11555911982149579, 0.09946449730950337, 0.09544678707582455, 0.12425676574369252, 0.12914049799955948, 0.14167980147630205, 0.12182982383327663, 0.08255357621795659, 0.08369497497687174, 0.09270524352869125, 0.09307711788285536, 0.09319111447786975, 0.10053501461924154, 0.10938481475267572, 0.09463522849511086, 0.11139019608985865, 0.10454092363151898, 0.12221799672483048, 0.13063721825255883, 0.10616900922786165, 0.14098147333431216, 0.14696586221470118, 0.14669049461009587, 0.1495926537146095, 0.16509886426455483, 0.17776674715464383, 0.12050294057751808, 0.12112045690580409, 0.10917402783178225, 0.11389584949384636, 0.11744508815055164, 0.12679060254690622, 0.1374059147855828, 0.1365859612086563, 0.1593171057547643, 0.2054103662452579, 0.21949087503649592, 0.19239310611584512, 0.2025753163960419, 0.19228808417271084, 0.13021125638544337, 0.1254642710514813, 0.13094668885974625, 0.1357935710423686, 0.12521023638393647, 0.10510208416459862, 0.11241943867950052, 0.11958880540098865, 0.11880703847689059, 0.1114121453183173, 0.11315862654775936, 0.1182249813750586, 0.1222446861720744, 0.10268710263993079, 0.09695498956168062, 0.10252151962542563, 0.12680614835903056, 0.08547525603814278, 0.12339857554906612, 0.1078918445470173, 0.09959476270088422, 0.09471187981934913, 0.09321541038230045, 0.10872409988812931, 0.0891116652900517, 0.10589216147323229, 0.09952034277532328, 0.09409327374741396, 0.08892206116570897, 0.11055672354292338, 0.1000398818419432, 0.09264528315225551, 0.09817913368773419, 0.09968656344621737, 0.08117185409447736, 0.08564506577273898, 0.08444740517178596, 0.09604561944493718, 0.10862026392112382, 0.0743432734454821, 0.10150148957040986, 0.09632527967382642, 0.1004816516390293, 0.0893470679295427, 0.09294189603278087, 0.07955070422184501, 0.0897202823803337, 0.0788905896115696, 0.10060992011204896, 0.09219869548214163, 0.09778128124853645, 0.08345942850555278, 0.08690339605039699, 0.09091110221364883, 0.09692139665725431, 0.09981765666812563, 0.09405595101387092, 0.0899121721794457, 0.08547343497340341, 0.1032970941759675, 0.09860511825058677, 0.0799981039735339, 0.09307329902746454, 0.0992866364484296, 0.09065457466594742, 0.08683012621575403, 0.08893022629307606, 0.08482100272520031, 0.09180904082005258, 0.08523352314374813, 0.11231843428723237, 0.09678999954139855, 0.07932388358286763, 0.0960801135442952, 0.09190382216029755, 0.09293153702838453, 0.09613121401541298, 0.08341302855538571, 0.10775893762192022, 0.09009285432963635, 0.09166742482280575, 0.08210949057235802, 0.09687322345484203, 0.08891212813088696, 0.10022983473940394, 0.08991242974308479, 0.07077144110714047, 0.09244565005278184, 0.10364511959294609, 0.11937523189120681, 0.10131327051427132, 0.08824548446609808, 0.1160929940952508, 0.09067700410823466, 0.09957220641630991, 0.12642175244776538, 0.10664282858936801, 0.09841997462097346, 0.12148424130102423, 0.10411793542589437, 0.10760153626035672, 0.0958670810667872, 0.10051446905642702, 0.11856206308490412, 0.09689074878663952, 0.11080828405885329, 0.11404418257451834, 0.15655351554048635, 0.16158844405411277, 0.1817655207515531, 0.15735921983891987, 0.15361516605150835, 0.15395348479147739, 0.18319014688793003, 0.15908615442232987,
#     0.1452245792286791, 0.1317518759316013, 0.14677613442040016, 0.09485645505689567, 0.11458479275359187, 0.11242972808234611, 0.1239126629994734, 0.15678221236059173, 0.10980535583922856, 0.14651419582544079, 0.16369118926753617, 0.13863582905080102, 0.13933434153881197, 0.13138819385266343, 0.1507205530827814, 0.1652103390331279, 0.1894146736973546, 0.1579628467809397, 0.1292943425909681, 0.18662440172193567, 0.14375626383916057, 0.1496413603269034, 0.13498996870940613, 0.20094192310345863, 0.15181992325948057, 0.15684801697945233, 0.18878640560543725, 0.13742862857588392, 0.11966948887578761, 0.13320505380136546, 0.12393230786477072, 0.11213437355192121, 0.10401518787194977, 0.10024651064412453, 0.13943741580560784, 0.12254026016430318, 0.17154293257070544, 0.18606907738350034, 0.13845003840623338, 0.16178619778168482, 0.15941558733350605, 0.14309029567248202, 0.11685843289541883, 0.10106176992531958, 0.09341846431031829, 0.09548055917373642, 0.09761627741567239, 0.10358910110880551, 0.0884334227219487, 0.12354485063933195, 0.10664919241437684, 0.1199356593062495, 0.12399603576444157, 0.11607349395620645, 0.1289275237053041, 0.09273745221934768, 0.09578197871278098, 0.09845469793640377, 0.11954485377531401, 0.12052102672889085, 0.11500801141023526, 0.137131678605296, 0.13608275769560368, 0.14858481925310274, 0.17346274810194517, 0.14813312819547675, 0.14861188540968, 0.14112684607584555, 0.12288090301194492, 0.14127949254573813, 0.13056546273859868, 0.1163139713401907, 0.1379687192350836, 0.11649048342086156, 0.12133243215922394, 0.13442552169364771, 0.13701782915260352, 0.10868315461715342, 0.12005868162802165, 0.11910214843570453, 0.11177363498886751, 0.1200531882138796, 0.09273170778943239, 0.11473444755474914, 0.09988552248246857, 0.09872013849333139, 0.10582493619310598, 0.12359881883536161, 0.09130992826204512, 0.11070404878791919, 0.1305857877565179, 0.11937075836111649, 0.12703894845034996, 0.12595472790467604, 0.14181085614949934, 0.09696905406444006, 0.11781726501938199, 0.12353953717762792, 0.11299804257341627, 0.1146263220114518, 0.12142565431591434, 0.13986490937437937, 0.11820316712418776, 0.11196416038158494, 0.12554057584541414, 0.11742962644694248, 0.09247285098032247, 0.11000496386563353, 0.11021024642309078, 0.12905427704959024, 0.1118489926714281, 0.11887311064462332, 0.1105358992878407, 0.14086941299899705, 0.10198043841624213, 0.08850585753477344, 0.1340914983264486, 0.10115279107176801, 0.08695828594785157, 0.09453584401048767, 0.08850703321298631, 0.08976875578698969, 0.09316200303295583, 0.09677108652837033, 0.10121490560728123, 0.07782364277981603, 0.09409495737955258, 0.0905753974246954, 0.09743955686902592, 0.11112657446838095, 0.09335706647631868, 0.10330273777657084, 0.12039382832052073, 0.09839877116943438, 0.1113400588182752, 0.13324469136973752, 0.1125415622833354, 0.12138950889992439, 0.11260608635082575, 0.10828162601433526, 0.11587100334498966, 0.13467214081394074, 0.10962998741554651, 0.12518459011208943, 0.1069849273463143, 0.13977099855433767, 0.17480525975755645, 0.144295493454911, 0.1191250163645361, 0.1286486418291752, 0.1645111100577288, 0.13814806147853498, 0.12386696783879962, 0.14583569367170102, 0.13462962165035125, 0.12428189260358573, 0.10934996852898561, 0.12205967579874889, 0.128750166117564, 0.14727259234296408, 0.12629245735193775, 0.12150040851665937, 0.10758364445776628, 0.12779961759353686, 0.14896626103786315, 0.1902882360878363, 0.19646610143859575, 0.17356107786902555, 0.11265299774792521, 0.1471130201965259, 0.1369414212869782, 0.11659629674700922, 0.10199929363111374, 0.07878726754082341, 0.10291329838111332, 0.1122432332203809, 0.11576601899005147, 0.12389683741448293, 0.12829465560625614, 0.13915302328384604, 0.14647434800632178, 0.17484439817265468, 0.18874139439640747, 0.17521764883899113, 0.14737308384604142, 0.1491848238426044, 0.12535499903548789, 0.1117507849341335, 0.11612849624636236, 0.11342510626245975, 0.09903756072257909, 0.14774348895839384, 0.1293442639932981, 0.1354768477520679, 0.1460742793837948, 0.13941382939158883, 0.15480855232270727, 0.13609241707510572, 0.09464204559506542, 0.10235336974453445, 0.09422372962233211, 0.10495127196819526, 0.11180937082452912, 0.09746717610686977, 0.08667837551252268, 0.09504046140042771, 0.12222281906846941, 0.11833215903879944, 0.11528177723398274, 0.12759718258184694, 0.14124088331156726, 0.11329053344103555, 0.1334354610419455, 0.12269970951343233, 0.12774586937951143, 0.1327996742684264, 0.147535591209873, 0.13279183325579655, 0.12008621355520091, 0.1303184425116664, 0.11293401014165967, 0.11216066881861432, 0.09169993716729205, 0.10476308207049764, 0.09365309753920921, 0.10332446609337421, 0.10040820016858296, 0.09962924545251292, 0.11520720913504892, 0.1053340948773077, 0.10370326389953255, 0.10449592721985222, 0.10074775587868741, 0.09491416412211831, 0.0946119453145528, 0.08174764885066843, 0.092143258756518, 0.09161932007459359, 0.10615205091954193, 0.10863585942888247, 0.11505389473083728, 0.09445501044202484, 0.09873827267002606, 0.09806693921757158, 0.11263203944942579, 0.11884028820406607, 0.14214355519293626, 0.10653580193512648, 0.10314003581157753, 0.11426359855067503, 0.11649179784900136, 0.11360676630171285, 0.10016664491648751, 0.10232252562131362, 0.10459112516396436, 0.09352087547748114, 0.08981531360475747, 0.0993257674967279, 0.10190617205800788, 0.11376133002466991, 0.10352080414042193, 0.11847036207146634, 0.12476907654770875, 0.1444110868751764, 0.11120657682602847, 0.13251986378039113, 0.1299483152018379, 0.1073946278163316, 0.09121987875100997, 0.10083052485911909, 0.10927284410944446, 0.11778914888949796, 0.0961068008694257, 0.1020466936332109, 0.11192279286418479, 0.10232556492492727, 0.10008023479082188, 0.099947646735612, 0.10680421772124618, 0.09738878227492599, 0.11067475392806479, 0.09728886949783955, 0.11213752975651549, 0.10402332062570945, 0.11659667871533985, 0.10780863852865263, 0.10863085805190843, 0.10075643968492846, 0.11389812604808541, 0.09367108422031492, 0.08677214003741861, 0.10453882537799387, 0.11250967032899915, 0.11071265243567084, 0.1073593817140338, 0.14163203749904946, 0.14889442161366073, 0.1402289620547157, 0.16311950837990488, 0.17465585289222488, 0.1797975393388701, 0.159613555770643, 0.16001578198875363, 0.11739218679327092, 0.14996201719346433, 0.11613503113494142, 0.12142319907735509, 0.11766409957021548, 0.11631518304798422, 0.1173616103613612, 0.12349293302891634, 0.1272055750672258, 0.15203792628391938, 0.13856074731804213, 0.10178856624508255, 0.1165103459623741, 0.08971914401237557, 0.08920609463309273, 0.0965783894106301, 0.10886052372953545, 0.10387462628986219, 0.16535964886566934, 0.0926527543973266, 0.09005599281655315, 0.10723298108068897, 0.1231164559136537, 0.1124442640373067, 0.11635229193039182, 0.11173088506273479, 0.1298766768766713, 0.09807282673020966, 0.09696977612795775, 0.10512035738384529, 0.09793886362015417, 0.10247269714723212, 0.11284567587306091, 0.11291453856743033, 0.10593450024567298, 0.10775413624551933, 0.10653363345733109, 0.11196982784754675, 0.10329576318093729, 0.10332315256009517, 0.11875817766409871, 0.1301681719722182, 0.11070782335764046, 0.12045223344339868, 0.1196022011012727, 0.1578102831394722, 0.13616822578338142, 0.0895809377943381, 0.08555509647691358, 0.08066803842578242, 0.08010614818387062, 0.09742651917850588, 0.09478102055193524, 0.10091396311568708, 0.09122491555635216, 0.10434339058694743, 0.09810966599224995, 0.11064879972655413, 0.11778015069295408, 0.11317172194719392, 0.12324164010749995, 0.16254304833715114, 0.16640624529610284, 0.1371174004374086, 0.14537595674339376, 0.13007702348304429, 0.13338119276627267, 0.08748004597716331, 0.10852948035409829, 0.09944240098974573, 0.13072978501748175, 0.14793797043599988, 0.13965057080055152, 0.1519584144555375, 0.14740743798075032, 0.21224537141888875, 0.24114875784686235, 0.21847749968632754, 0.16922582469373634, 0.17851870084215665, 0.1391537055159765, 0.15332000371984567, 0.12557645694349884, 0.10588326329725273, 0.11025523563020266, 0.11115715866329173, 0.10592182975711269, 0.11484827858855322, 0.09662973560670012, 0.10163095008599722, 0.0891690564158722, 0.10365257626623338, 0.12906478626737689, 0.10939499232494368, 0.10298904447522526, 0.10548699656991659, 0.10532180821501265, 0.10321073358253936, 0.09757525425853163, 0.10805051356524736, 0.10830056150754161, 0.10580588908770265, 0.10567451930461391, 0.11669096880645675, 0.09889599557069445, 0.08951719673178475, 0.0995106915051053, 0.10347599362885716, 0.10329475688256144, 0.11399676844680377, 0.08100571304367979, 0.10505880269069456, 0.08816323080488388, 0.11133294326222014, 0.1011241447889336, 0.09251799422396835, 0.08272470126393101, 0.09013508774594282, 0.08826014350245436, 0.09285738892759368, 0.10399653063261309, 0.08531166874789849, 0.0885856585046575, 0.09965863914068729, 0.09732319992302152, 0.08821179025483622, 0.09316961532685931, 0.09687633745840428, 0.0851772790141392, 0.09606761529440157, 0.08332634497338644, 0.09562512986071474, 0.09818359521780866, 0.07901613740085858, 0.09277948499593525, 0.0950769328970112, 0.11420405028506325, 0.10888378186170762, 0.0904689239505418, 0.10849847496786263, 0.08998118046227899, 0.0800306070706515, 0.104191995083601, 0.09067911632070241, 0.08577676479267186, 0.09931071361392477, 0.09792368573313757, 0.10367525944877155, 0.08318290680354963, 0.08174363006587367, 0.07883357199076921, 0.09514115903290304, 0.09540891445224683, 0.10614255571703284, 0.10152942809817825, 0.08431386690539287, 0.09808128728146405, 0.08007254754210336, 0.09675594224387829, 0.08570283877383361, 0.10023233959514723, 0.09311099523657244, 0.08591261870912129, 0.08712729426578508, 0.09425493452298764, 0.08833378827892104, 0.08598671188999422, 0.09640576603844828,
#     0.1159032078445344, 0.10006484492467861, 0.10491802863491503, 0.11278646491091532, 0.10506196994388205, 0.10532415818477024, 0.13047890872340304, 0.1307235535852491, 0.10832651075577575, 0.10589909515536057, 0.11021471749640145, 0.1109628154935667, 0.11340998512896434, 0.10310704368421235, 0.10275997358712548, 0.10937426845195529, 0.09987150194876601, 0.1370815037256429, 0.129461528960889, 0.1510774903921523, 0.17622425244348502, 0.184437439218602, 0.15971852193334526, 0.1810531934026446, 0.18363572850314294, 0.17731475505430566, 0.16111955944069967, 0.1371650536334075, 0.14668190360243022, 0.1307064133267375, 0.12289119214453102, 0.13832017113577794, 0.13713605665691225, 0.12096345837309988, 0.1378192714977005, 0.12948764770046703, 0.1272301188190119, 0.14240645745930083, 0.1708029012397826, 0.18197609232318124, 0.16022477692394316, 0.16078093805940885, 0.18958898415822853, 0.16975129331661817, 0.14373244603457, 0.13041475195188154, 0.1454230138412856, 0.14520174448702242, 0.13082708606575333, 0.16958490199985893, 0.12494670125317844, 0.12875437045095445, 0.1442628060225945, 0.10935238822767734, 0.10481906793956859, 0.09968672386996301, 0.09677961981668987, 0.10372202393779381, 0.13125118510838532, 0.15804323671380008, 0.1509647922199873, 0.15696368082711473, 0.1243642416659681, 0.12502278195780933, 0.14026625230093737, 0.08967018842583543, 0.10613992955461062, 0.08591338515370883, 0.08692453312400408, 0.10873518440842392, 0.1173309569233087, 0.11910399814414602, 0.11323370932161263, 0.0978285480551609, 0.123934078693749, 0.11760718779829318, 0.10723570304824666, 0.11684090090633172, 0.091044027542517, 0.11534098509719175, 0.12668514657854774, 0.1300563279798545, 0.11064772882699488, 0.10427922414824298, 0.14856850177629094, 0.13286956951245765, 0.16818349453109868, 0.16347333790706467, 0.14770237041759296, 0.14606313198117057, 0.13224105085374435, 0.1278430760193474, 0.14028047809093835, 0.12326935229347094, 0.11357545080937603, 0.15578689271283852, 0.12150401236077085, 0.12402288256255473, 0.11876670275117929, 0.10777786188994334, 0.11867196828635349, 0.11903081481953591, 0.12772000594096497, 0.12707774664959012, 0.1100470981176968, 0.10943060921204627, 0.11233655149437982, 0.09589574084802896, 0.10185880385327886, 0.13554944951181852, 0.08905658187512479, 0.09852382002572405, 0.10804926534935125, 0.11604446238655557, 0.11023577892262243, 0.10334097944321556, 0.11473225009657885, 0.1118569856909221, 0.13455863530354248, 0.13847406467190077, 0.13650472364388663, 0.1190701315505584, 0.14147640504455106, 0.15558779121352856, 0.09983984766952178, 0.10814476446619198, 0.10291509880624268, 0.10637887935570556, 0.13259451056586288, 0.12227128444220701, 0.14015325405054813, 0.1329398130248097, 0.12349159658231819, 0.10932174898739425, 0.12537588756445017, 0.11927295020675976, 0.12859235385486242, 0.07984816856163743, 0.09456430096332777, 0.12765197001245188, 0.11747404765254252, 0.10286389822843327, 0.0793479563728957, 0.08327569436521845, 0.09293195232467619, 0.09726171538934898, 0.08318206547387734, 0.09954090103273697, 0.09046963836099861, 0.09674260194192881, 0.07929560966331516, 0.12207396605022959, 0.10013639854907237, 0.1025615587503767, 0.10808772219224323, 0.10560884558813777, 0.10863537517422159, 0.10624861965357131, 0.11033695816830438, 0.11314226619656612, 0.12926010259716647, 0.13909805301264105, 0.10342755024706377, 0.09613713636703189, 0.1275865548892425, 0.09993767375957223, 0.11706609040358458, 0.10670035341685258, 0.11783993861368242, 0.1344725188239459, 0.1636676956141197, 0.12188500193366113, 0.12621951233926723, 0.16287682396437766, 0.14370919666712992, 0.12609051265156565, 0.12670263667782292, 0.13517746316722648, 0.13321461239902854, 0.12474669526797727, 0.12446460690874443, 0.1657972983547399, 0.11879904766430437, 0.1219971718260039, 0.11454870025430984, 0.12629195559505102, 0.1389427460649328, 0.1442127845054276, 0.19129512311564242, 0.17595032535962812, 0.15710239881412866, 0.1351728389959294, 0.13758245514885664, 0.11706992438482666, 0.09886004819127199, 0.10508217222021947, 0.0861718379582024, 0.10706648171288442, 0.1102651601684748, 0.10137464208151842, 0.09296545807770458, 0.12351800472508355, 0.139141637771713, 0.1489648590835933, 0.2030346088930872, 0.2007674365082366, 0.15146108071423572, 0.13947604849819095, 0.1472820317864932, 0.1291553753207961, 0.12514170198256397, 0.09756023617807373, 0.12477703897696751, 0.11712877913357794, 0.10696504878232053, 0.1405834314178227, 0.1278676806347309, 0.14409221284167942, 0.13023907601752704, 0.14635589852272385, 0.11993179606794363, 0.11796784147631079, 0.1070867869460118, 0.10185169756493756, 0.10072386416162593, 0.08664208028784015, 0.09664096229976783, 0.08386985558508815, 0.10928554925376105, 0.10241056512121299, 0.10574206898243481, 0.11059340136484677, 0.111574576000467, 0.124166487059878, 0.12177514695603013, 0.1381979444564203, 0.11660932180795716, 0.09942697967163476, 0.12931280033360631, 0.13857713559163654, 0.09987940934711867, 0.13188158385868606, 0.11693970282379829, 0.12291307258492103, 0.12517265433975103, 0.12353504852669218, 0.09577678324646244, 0.10069388447970126, 0.10188483884821385, 0.09928960273409777, 0.09679910050337012, 0.0931203189283031, 0.10644896547131096, 0.08746545300433764, 0.10268063463157125, 0.07974211769124107, 0.08583193912343845, 0.08791270543647141, 0.10183265739212487, 0.08710729617078238, 0.08682413275008245, 0.09694961718810893, 0.09306936229703248, 0.10924006498572707, 0.09383832205366494, 0.11546858079655387, 0.10817282447222522, 0.12864725371233837, 0.1088958416447556, 0.10083803395468731, 0.13556342439554198, 0.11350721968856232, 0.12605067357202293, 0.10914482692929289, 0.09184159705287194, 0.09903659185368456, 0.0942091998610045, 0.0891030551720202, 0.09380189961417425, 0.08285793247840217, 0.08090014647808942, 0.09635972140949223, 0.09156547410094364, 0.10426127982859155, 0.1059608306646032, 0.11889120232905856, 0.13603207655371577, 0.111873692547144, 0.12041506259319065, 0.11628527649330332, 0.10473311411292585, 0.10928366519401984, 0.10453031412007815, 0.09244679255724726, 0.10443302498733685, 0.10399055932297592, 0.12979530152454835, 0.12123297740534453, 0.08952701835435524, 0.10923268931479065, 0.11255536286254685, 0.09242982936463966, 0.09612738778051551, 0.12350107845184807, 0.1145593945777888, 0.1115495886723755, 0.11171966208864571, 0.1247048215692057, 0.11418183866765245, 0.09390413111340118, 0.09757981341289258, 0.0900905107882869, 0.10473944252893037, 0.11094128155669816, 0.10295800632667951, 0.10474232309398075, 0.1086280833249523, 0.11125876837676146, 0.1480679897706517, 0.14425314915981363, 0.14119535449635837, 0.13786089039449487, 0.14330351773552466, 0.16239163780562096, 0.15074691492263778, 0.15301779721585454, 0.1427570642113311, 0.13740294514674853, 0.1379733623609134, 0.11491109377639566, 0.10439332648387208, 0.11409682753925297, 0.13890583104786794, 0.151175357941238, 0.14770607770939884, 0.14887614313976624, 0.14486877976632184, 0.12270895056321153, 0.12962879185212492, 0.09739265231602827, 0.10428599432665578, 0.09316293636540365, 0.11556983816272505, 0.1262457044847325, 0.12938845559478532, 0.12054630263794605, 0.10721079481119919, 0.10633788492506668, 0.10194323082936084, 0.10399170945333293, 0.11193097204287081, 0.10776278150296552, 0.1442067237822199, 0.09708382202921394, 0.09431475172072284, 0.09650708190244922, 0.09525150371186988, 0.09294221507302672, 0.11357419126323463, 0.12184161910155925, 0.11087114135269557, 0.10378644897320707, 0.10799831275675172, 0.0961346338856277, 0.11153137611072218, 0.10969761403726543, 0.10931630442194623, 0.08833622829495512, 0.11705665523330006, 0.1426191680013788, 0.11199706970883394, 0.13870670492287993, 0.12507040918851794, 0.09536745347380873, 0.09416819530290352, 0.09410468174903064, 0.08941781046738909, 0.08558264981664077, 0.08159878735418906, 0.08958522598164798, 0.08370543860475764, 0.1103560777003099, 0.09720060754951053, 0.1014017383656509, 0.10327610543909709, 0.12970818426282651, 0.1319104140813978, 0.14560211651893332, 0.16593868310731894, 0.12770273426956047, 0.150466277357794, 0.1507482368954221, 0.13504515157294883, 0.10219282228806124, 0.10288530852832396, 0.12346612411671391, 0.10473888843649767, 0.14314847923566362, 0.14639637780968515, 0.16990590262783037, 0.15015438904783868, 0.23652008402626115, 0.2159059197869406, 0.19858099142003524, 0.16324620320581323, 0.16238001970997717, 0.14326008098060808, 0.16729509269592696, 0.11374470741766349, 0.11851222073948257, 0.11900322998213629, 0.11214142143074766, 0.09773348925153326, 0.10858024002625236, 0.11658053797993662, 0.12575315766115894, 0.09633739629359947, 0.1174160189126379, 0.10425780434820045, 0.1106386561629193, 0.10414398759726502, 0.10841869112935953, 0.09341267701476803, 0.09012058659405002, 0.10553643947840528, 0.09916585314471311, 0.09703404069554807, 0.10338588325628253, 0.11496726251147081, 0.11186730725523672, 0.09087429969059023, 0.08319149211882283, 0.11427696448224861, 0.08931286414512021, 0.10436352414944408, 0.10375779676586777, 0.10773369947710403, 0.1122949265209705, 0.09115281711537485, 0.10104355937163152, 0.08746447836840586, 0.10648574629975743, 0.10465338457925932, 0.08238797595464559, 0.08300129062905127, 0.08338749576006933, 0.08313446816274796, 0.08272367756980473, 0.09069482512113182, 0.09615733949563948, 0.10317585180095336, 0.09395590904849124, 0.07899236579100456, 0.09754663600288926, 0.09271534383285131, 0.10328670903264757, 0.08306962108125453, 0.10606926433203938, 0.09480862531684263, 0.08590496756852767, 0.10178312830278122, 0.10446646466709145, 0.09750099213087812, 0.09519799505244828, 0.09379913791215765, 0.08340052188421727, 0.10089537552483417, 0.09562163167080225, 0.10017170755110485, 0.09721999020802821, 0.09781295831042026,
#     0.09448935405582874, 0.09998417394382869, 0.08600162526668652, 0.0970907655422024, 0.08032979926536211, 0.10191224915319454, 0.10315416636383819, 0.08200753442934704, 0.08513868774015326, 0.08320853687185858, 0.0767082260944751, 0.08256358513167242, 0.09033686921966197, 0.09123490380036602, 0.0722707250348406, 0.08234244459209673, 0.08162683470624034, 0.08669604323114088, 0.10461704522320195, 0.09231712496769699, 0.07548825601194624, 0.10121029287905176, 0.09090898532889143, 0.09301679099320047, 0.13051536207499134, 0.10271703723579911, 0.11173983132342794, 0.10524927966983547, 0.10289416947933187, 0.1052382793654002, 0.14086852906232328, 0.10402870894425642, 0.12358536171642152, 0.11044662321239267, 0.12154510865921346, 0.1202549209121976, 0.11372663244429379, 0.0852855931588871, 0.09556165889845422, 0.11184416263197063, 0.1223360351539461, 0.1434140887199811, 0.13647707126468464, 0.1483164651398912, 0.19647919081658258, 0.14713457080629408, 0.17870529230719026, 0.18298491132200642, 0.16685518944476663, 0.13340976035518307, 0.13297683498574886, 0.142040940794806, 0.1407048963262613, 0.11475111056511171, 0.11412841292159212, 0.12204839045505386, 0.14334259769769678, 0.15133547611145048, 0.09899640289883943, 0.12004912881592278, 0.13826900109370785, 0.1483753808169774, 0.15554068524395767, 0.173980325390691, 0.12395485687643003, 0.17532467892139447, 0.18054344677881098, 0.16152547274343243, 0.15941524138651514, 0.15681063077912155, 0.14344409119859539, 0.1467654158903334, 0.15631153197496372]

# flu125 = [0.17558776712673363, 0.18232031683793049, 0.1763325679889482, 0.18108667775084925, 0.13657423450707235, 0.15019480260340345, 0.1379424285132922, 0.15424134666710143, 0.16867345200544026, 0.19222754649372784, 0.20561366804196826, 0.16929909773117818, 0.18205009524446822, 0.1693645659501694, 0.1920479240497563, 0.15133920561864891, 0.14472370425300857, 0.12297353627972911, 0.10904657166928035, 0.13833959661443954, 0.15507818722793335, 0.17118264178077083, 0.19375117206235318, 0.14870880585472296, 0.1627100388189232, 0.1416715226070076, 0.169516104921026, 0.16614358990149258, 0.11969611309542373, 0.16164372912571448, 0.1369479006544577, 0.15273944319609303, 0.15450133822622789, 0.18683358951631768, 0.23547606359095102, 0.19544196236156564, 0.15510164895907688, 0.19319228468384694, 0.19952209238364954, 0.19001855080454388, 0.13489125099545357, 0.17189649778605676, 0.16530076258167356, 0.16419835505013197, 0.16695033743958893, 0.21039871307510036, 0.17385059973054243, 0.1507724755227426, 0.1819128858729891, 0.1505950444600248, 0.20716430224091048, 0.18402750367060586, 0.19181919693998611, 0.19987592285258252, 0.1602595612772949, 0.15831751031022925, 0.13308114832899473, 0.15643171529251032, 0.17608687673222165, 0.14856472527298886, 0.15801340682556322, 0.176108254191601, 0.13523089359017834, 0.1382272090180814, 0.1961765606743724, 0.22208565023032062, 0.1508575522981815, 0.17246068199598896, 0.16485030932630615, 0.178701089406662, 0.16750086556694477, 0.1843100206771919, 0.2068286897699085, 0.15379891556654052, 0.1852426319797187, 0.17635117872736214, 0.17833933987115813, 0.14927467785046344, 0.14768802177797163, 0.1685941518983459, 0.19408437745860102, 0.16387727980445219, 0.16600488992804216, 0.19538819106517982, 0.18158271996541278, 0.1969864367374116, 0.19339329186299337, 0.1659885762055939, 0.16273402362305142, 0.1441789128278119, 0.16361963145501582, 0.13835659617572973, 0.1505796129545575, 0.1104657550515818, 0.1362199401107387, 0.11670411045855822, 0.1283696800272659, 0.15192606456926996, 0.15746063567831137, 0.15045679056374103, 0.13499780080096943, 0.1384014465405101, 0.1185558599659119, 0.14247616654619605, 0.15490955977754486, 0.12587769348702277, 0.13791474293514933, 0.1794378951033699, 0.17732449994927027, 0.1574468582567394, 0.1503935644330959, 0.15842855352037383, 0.1893552419844495, 0.17180619921755683, 0.18105106651293212, 0.1677913820135802, 0.14172107043075702, 0.1982425257934525, 0.14847429822836153, 0.18030568174967299, 0.2521342624577487, 0.19531161541629904, 0.210759946927382, 0.25106472133456575, 0.24874046623079668, 0.16172761452520487, 0.16299724224263543, 0.17316586296346742, 0.164563826811876, 0.20478025793035223, 0.1907165705406043, 0.16139691254142108, 0.1792046450138125, 0.18398081827235505, 0.16417698278145149, 0.16921230196874387, 0.2124492674158381, 0.21571598769571365, 0.2683845210706441, 0.25570529551814264, 0.22159653017716874, 0.21376949639031237, 0.18502209477221443, 0.15075646156434402, 0.12958842262091647, 0.1644661441916888, 0.1622734664704234, 0.16848846517812135, 0.16548023038008294, 0.15604066500867259, 0.16584016029357984, 0.17595473189645935, 0.18583939310302147, 0.20069297007224013, 0.24714739090997093, 0.2485848891277027, 0.20581805134761863, 0.1921133647946643, 0.21613637387511392, 0.1554337445213453, 0.14995855149860604, 0.15411085619070114, 0.1676914529592658, 0.1614812829655531, 0.14815828602327993, 0.165459502457049, 0.17089196222835915, 0.21958877854263456, 0.1940167956786712, 0.2453540079944958, 0.15163958781631362, 0.13033791424457442, 0.17238939461707053, 0.1449893447162341, 0.14287948298795236, 0.17751939018201726, 0.12978033620108556, 0.13293017923761877, 0.11897332097082368, 0.1395828353600158, 0.12054094052260944, 0.18342639666863764, 0.16042154235310535, 0.19028400002384951, 0.18176599656408438, 0.21043022112906154, 0.16869799780475356, 0.15692760890881954, 0.15791399492130012, 0.15905421137770356, 0.18137622800451905, 0.1594502505471108, 0.16743294155313107, 0.20047320282364509, 0.1641125451853691, 0.1472136393093473, 0.15876707563349518, 0.12703041182752423, 0.13562823903547083, 0.13962552943201442, 0.1347731661614811, 0.13031744176812107, 0.13805791652616634, 0.14990997151485191, 0.12157445447909623, 0.12140194007712114, 0.1475075319339381, 0.12390241819950149, 0.12116103190576082, 0.1301167692413937, 0.13073181906212603, 0.14352544026614966, 0.1614064906980462, 0.18803467326606568, 0.13770176462377318, 0.15024207374921328, 0.13035273828641677, 0.18708243175936054, 0.16698722259145976, 0.1378542346164905, 0.16471891678368814, 0.14231952475757326, 0.12908470732284502, 0.14748658200448714, 0.1522453098580441, 0.11722263352895566, 0.12846098523892718, 0.14315090613679496, 0.14159105467295932, 0.10948133443195206, 0.14504971842020234, 0.1299798816295582, 0.12649566598042666, 0.13336143948049142, 0.15877127059418208, 0.200522696334992, 0.18392610669098478, 0.15179974480465375, 0.15019219534897277, 0.1677100685968056, 0.13663606249099614, 0.13067198893297804, 0.15926957457918636, 0.13262083161520327, 0.1360876483195053, 0.1355365590462908, 0.14212823209299222, 0.13010293047691693, 0.13558701851841493, 0.15697869059028885, 0.1505154469263578, 0.1813853622432273, 0.16435188343433252, 0.14126885743279946, 0.16309974205223432, 0.14580531009359474, 0.16113519905237714, 0.14693281172577713, 0.16191308699947454, 0.1552299037319835, 0.1363330965605703, 0.151801277982432, 0.15206752745088037, 0.12819713583147674, 0.18232266395253943, 0.14955530947955042, 0.15927416143756234, 0.16554616862218383, 0.2011024310413941, 0.18590182807900396, 0.2148132486286766, 0.23553539422391714, 0.1810911850857181, 0.22311310768708809, 0.21871419392317268, 0.19595458732589532, 0.20655352174777772, 0.15690482951413517, 0.13493542403829978, 0.16452297843465744, 0.1513604635571188, 0.16878742171921074, 0.1523613130095815, 0.18768049506923412, 0.1761371748024313, 0.15473070282841145, 0.17287325710412216, 0.14560630579669478, 0.15171474430855492, 0.15136084386067908, 0.13188418364411175, 0.15878558090221775, 0.14840765521649255, 0.13556807789166128, 0.1829052393299389, 0.16751113195367393, 0.14053798699366632, 0.11960466715366319, 0.15603353237049264, 0.1570233757689478, 0.15609963794767634, 0.14633213350075774, 0.12769561651113506, 0.1575161449443658, 0.15035968863041785, 0.15273519465497692, 0.14620932458411298, 0.15727868067215395, 0.17065713798281024, 0.1373573835808041, 0.14780704363356126, 0.14070174102223162, 0.15177381758939026, 0.12701633551968095, 0.1400821250474674, 0.14539318166956142, 0.14228854170593816, 0.13409397791993782, 0.13845808567551837, 0.17880424991556906, 0.16459060855026322, 0.1602315186610774, 0.22222310598649891, 0.12392002756237434, 0.13171752412185908, 0.10769458318172558, 0.11223589871749401, 0.10211001368616573, 0.1691800269190372, 0.1672160929637412, 0.12205155153801849, 0.13511393996564836, 0.12227634455447121, 0.14819849771289562, 0.15060539796209343, 0.14326740252144296, 0.16460990323458144, 0.18743980686917638, 0.18965948535824367, 0.17513562510415667, 0.20563743042656463, 0.1841344789575294, 0.16043286782434568, 0.1389001691019449, 0.14178151498471106, 0.1440989505763469, 0.15163133536787438, 0.20294520424272275, 0.2060704984041227, 0.1794196142399462, 0.23278790070775848, 0.29489440405079914, 0.28592254672593653, 0.29601888443504887, 0.20818128446534834, 0.2317490927248466, 0.21931646196183807, 0.18669179982165313, 0.18179421574872917, 0.14381908716232658, 0.19513350194778534, 0.13222962308111932, 0.15699178024077054, 0.12926043477439916, 0.17625952260475095, 0.13782876106389846, 0.1376272017505605, 0.12308685475973273, 0.14041917852459498, 0.14341252481744654, 0.13448477582064933, 0.14642958244196164, 0.1507364936192939, 0.13806846681456025, 0.1418516617387365, 0.14486978395239494, 0.13993958883300597, 0.1222822283336181, 0.17123013271349255, 0.1400318347629493, 0.1437657155264609, 0.14324295527506023, 0.13451987403850402, 0.143100258960206, 0.12719276455183104, 0.1278208523379269, 0.15903135391574746, 0.15385142134057786, 0.1448984356983885, 0.15016962407209836, 0.15402903352988698, 0.13337830322627445, 0.11502307672452311, 0.12915145495002625, 0.11798275538002713, 0.11239257375293611, 0.12733317450853737, 0.1256327895453323, 0.12736049555063753, 0.09911395726362654, 0.11319699697232985, 0.11728524172605465, 0.12859997832455275, 0.10790432590371878, 0.14077530539850477, 0.11378568549529365, 0.1569104172401882, 0.12836064243156936, 0.11823281277656353, 0.13205546119823885, 0.09494230255925672, 0.14864143746196465, 0.13051175629887996, 0.12407066626855028, 0.15432616154429019, 0.12715999387516577, 0.14139543374381772, 0.12488272014198526, 0.13441094522037111, 0.13364407074479873, 0.11788024648195919, 0.10970716317239894, 0.12993123533684814, 0.1344112679200776, 0.12008302830055263, 0.128194360360653, 0.12940804291434108, 0.11176798694251455, 0.1239078693614328, 0.12417666195365842, 0.12428767116582896, 0.10904752477796872, 0.10356223960234685, 0.10559428258277968, 0.12107873971853719, 0.12655574003869302, 0.14826176417101392, 0.16187009139442313, 0.12331717349842301, 0.12099370609818651, 0.14132423978547892, 0.1323437075169557, 0.1324525784183191, 0.16092501884846408, 0.1224196931082672, 0.14511455217828428, 0.14646465290413066, 0.15448702696583605, 0.1518757422393379, 0.13657101940900646, 0.1708921638292596, 0.15544981731067684, 0.15016586867587467, 0.17531076352058705, 0.14163812329449732, 0.13799526790973052, 0.13101058745700625, 0.17027527744426396, 0.12499581985674052, 0.14954514788781859, 0.15279192347736545, 0.16339165776137546, 0.17802207357480146, 0.19216636961644545, 0.202086701289963, 0.23415289193900396, 0.21232865514211366, 0.22376124620876814, 0.2555226033338877, 0.22168264108549318, 0.23468153636341615, 0.21357815136289368,
# 0.23858648610639627, 0.18839818888063903, 0.1629337370889445, 0.18152870691470127, 0.14444806892154158, 0.1785268102902184, 0.17770306935424884, 0.15709507415842922, 0.19054238900730514, 0.17274555091083493, 0.1943787592637947, 0.2142816973675369, 0.19702637647626026, 0.18391963321232666, 0.2523168282507399, 0.23280011356109837, 0.17817334302934762, 0.229144696401671, 0.20130232014610755, 0.2055094013268968, 0.20467435053985483, 0.22170574081580935, 0.25210280632904036, 0.2318317372072347, 0.2059068349659469, 0.24645135844318056, 0.20009236209320355, 0.18932696110574485, 0.17881099688283078, 0.17796649819605398, 0.12749021149806092, 0.14617524554291178, 0.13980450279711715, 0.17772212367266846, 0.18535148477014124, 0.21481556272914082, 0.27698882652332735, 0.2277937202085335, 0.21937608084699237, 0.1512344194971648, 0.17740180362876937, 0.13689750221582064, 0.14077007168109096, 0.12268261644374077, 0.11019546345026422, 0.13196873758826905, 0.15936197228401114, 0.13388432956270374, 0.19613053717911869, 0.17228075863327447, 0.15675563624580724, 0.16521890885552237, 0.19403233089622507, 0.16072307209995923, 0.15706948240802596, 0.15466427919064898, 0.16661536694984355, 0.15739462732385068, 0.14918370895962263, 0.17839133129940624, 0.18549516923574, 0.18646007346456822, 0.20315081641111207, 0.24479956891311352, 0.17965361204571176, 0.18629025860241485, 0.18720872527484908, 0.18074376156968988, 0.21085017935869307, 0.15410670421890682, 0.16847412743235668, 0.19792725829379812, 0.19762490709096997, 0.1848560987826359, 0.14670986454541876, 0.16359331696342155, 0.18813751760823474, 0.14122814177245177, 0.1420292517212626, 0.1509816778851178, 0.15955159990355916, 0.13500059786276175, 0.1515157557864096, 0.13808470021794028, 0.14620825519790712, 0.12575296577666553, 0.14179386497487245, 0.16437057871221952, 0.14368205417819016, 0.13503499280703335, 0.1502645037984261, 0.15971328126834314, 0.15895762527810336, 0.14130768642336552, 0.1647659698368948, 0.1733201283277799, 0.1902569900014699, 0.1553591677838858, 0.17342475921240005, 0.19286854018547994, 0.1763814365918525, 0.17004810979921295, 0.1430217485791173, 0.1571599358385863, 0.13803847963050855, 0.144004025811556, 0.13797954559826445, 0.16230089275078688, 0.13264134159233446, 0.13498406784964728, 0.1358206754325518, 0.14231017103318122, 0.17701882826886584, 0.15528065198658478, 0.14951806336954696, 0.1530443711749624, 0.18413585110950287, 0.13631329000236325, 0.14096468616047336, 0.11457629613658837, 0.1230287574206391, 0.09577246542333408, 0.12087204023451767, 0.0997104901543136, 0.15611046971975365, 0.1288527318719656, 0.12904616152360904, 0.12022941378694524, 0.13793753643369497, 0.14968256014065723, 0.12492478216299994, 0.16230183436153903, 0.15626187341509737, 0.14863854192461248, 0.17428965947872405, 0.15440402137668022, 0.17011459254857672, 0.16828321232640472, 0.15768754504895385, 0.17569250785546936, 0.16249745066685478, 0.16579592371904203, 0.15312098857939901, 0.1588943691128713, 0.1576693692051485, 0.20676732185332847, 0.1962566694894313, 0.16456857165464076, 0.21082250011477136, 0.24067865086049225, 0.19721293198385456, 0.16975676562631525, 0.19689713360989541, 0.17736144093009293, 0.19830760440597728, 0.17924184152528536, 0.1666356106503349, 0.22613598609372149, 0.15025390894495538, 0.17335086887881646, 0.1744122035326345, 0.1798375949413832, 0.17362684798569503, 0.21045958295727196, 0.2616918676985744, 0.29442588443006745, 0.22094370897587165, 0.15785432571888702, 0.17901727158366526, 0.16990724940191718, 0.13525827515936903, 0.15888219719875266, 0.13869592755566648, 0.12421828728967925, 0.1376865822488824, 0.16913356008863492, 0.15337722768477846, 0.1525356836237962, 0.1684802406823438, 0.222044443732663, 0.25062398230759564, 0.3023544145761683, 0.2210106905003305, 0.16831574546461409, 0.1926087336770311, 0.2133988124585578, 0.15042343670301828, 0.15724434242350097, 0.16932458551725704, 0.1679578099073831, 0.14549492528775604, 0.18925525021802936, 0.20821207777635706, 0.20012350207236784, 0.19591174363015326, 0.18489238722262508, 0.17727659048143082, 0.14340193672335472, 0.13979374824968588, 0.13772838098461473, 0.13857685349260287, 0.16033788638505558, 0.12468206746406828, 0.13356583342285472, 0.14453080956413067, 0.12183429888730125, 0.13111675947768062, 0.1363670390520757, 0.16214479807747162, 0.15911732376488794, 0.1670320276514017, 0.19993483821169017, 0.18740113925717375, 0.1603284566066626, 0.179313697595493, 0.21623523498120026, 0.1737751465814928, 0.19128581572600695, 0.22094413792695514, 0.17648281267500904, 0.17610231393605175, 0.15405278842720077, 0.16613026996829894, 0.1365116214452379, 0.15840451325047078, 0.15930011792919566, 0.1456405340635384, 0.13659004572290112, 0.14916199065467461, 0.1303696612670413, 0.14487752487098313, 0.15818009084123172, 0.14765636714363764, 0.1460066580548377, 0.11356801260266114, 0.14110017979159498, 0.12698126068732235, 0.13604476506792057, 0.1491427270519194, 0.13149566550676953, 0.1407627765263863, 0.16606085149001928, 0.14980817556558443, 0.14381463846244577, 0.1437231652714167, 0.1291275095237927, 0.17826036347673754, 0.15611236216292323, 0.15189713059312446, 0.1486015733976211, 0.14000373806072924, 0.14497519623598254, 0.1263808518324999, 0.13965960005502354, 0.13058895368118562, 0.10802000078245708, 0.13595547789464094, 0.1319429788456204, 0.11945980518682332, 0.15478011412056777, 0.131136301937761, 0.15780477531721823, 0.14890958961586803, 0.15723929089243327, 0.16007004050504872, 0.15620458875550175, 0.16382505181601775, 0.14728774432421018, 0.1569573637221125, 0.15400646714271346, 0.1551154281830189, 0.161079528206243, 0.1371704488061354, 0.1282972411979981, 0.1248707284173128, 0.12473515959426738, 0.13162993764449796, 0.13670858480027212, 0.15136270194735868, 0.14807179873451715, 0.1423316081294455, 0.16689038209852197, 0.19512013000453232, 0.1877340525714653, 0.16854950342392785, 0.15096165722298455, 0.13239958989263823, 0.12352784106708642, 0.16973869144406867, 0.13712834826050352, 0.1549848595706064, 0.1608678587135826, 0.14917100580973855, 0.16286913874876283, 0.19545723985406288, 0.20159888334593096, 0.2190089238817066, 0.18005427887774317, 0.20336806733447324, 0.21366017163765944, 0.1967102551791775, 0.17018614633937054, 0.1992929204148804, 0.19132037409013264, 0.16507641466579406, 0.16554638961152726, 0.17772286818787783, 0.1445352242160966, 0.17565657911516425, 0.14530947883811635, 0.1584037854977961, 0.16085709422860253, 0.1524106947087808, 0.17924979074396738, 0.1619539595856269, 0.159904230335944, 0.15214654257183288, 0.1614912067579913, 0.14506812006459277, 0.14832130609639585, 0.15192224153548906, 0.14884673592970582, 0.1293591588005829, 0.12138213120427288, 0.1433395913546108, 0.13430309521490785, 0.1514988217303604, 0.15107512565637587, 0.12883981876053266, 0.13698395950847622, 0.15859889446021427, 0.16458040152729644, 0.12633564557691931, 0.18251834846212456, 0.16927182004295233, 0.12280404348491718, 0.13795874917726428, 0.1264770039365859, 0.13491687956671722, 0.1237273657439379, 0.16166395297557234, 0.17699957733400792, 0.1436934118896991, 0.12297746562786928, 0.1381458128993561, 0.1601905368246762, 0.16726417870042168, 0.1602075894472493, 0.19405086181615072, 0.13030206527493013, 0.1228978740695866, 0.12666446435260015, 0.10083050278250329, 0.12143696548260222, 0.1374160746099285, 0.14120394753044424, 0.1243574499609655, 0.14896021176702257, 0.12151869642752704, 0.14142732676736977, 0.14927166921696983, 0.16159492266745895, 0.16721862639184168, 0.21029256901202598, 0.20492841585572652, 0.17709673734072867, 0.16746777507610838, 0.17519705503601857, 0.15044514388741936, 0.1984304305826342, 0.14834039615874872, 0.16670739881433694, 0.164053885035737, 0.18346599936342947, 0.18552106116774192, 0.21483612910699829, 0.2033540431802833, 0.27057327910559514, 0.2621149637967984, 0.2626710104586713, 0.2770431499229195, 0.20689190950682024, 0.20866865146559652, 0.18187944994881275, 0.16151992044196872, 0.1453961625557677, 0.17379400951048116, 0.1448364249903647, 0.15746502094352113, 0.17237395431634592, 0.14707996999152814, 0.13131314703109315, 0.14520336114395296, 0.13764205542891564, 0.17562187416895209, 0.11696928320684276, 0.1464917865321462, 0.15781699873485402, 0.1438654201366117, 0.15268896753300373, 0.13538877443512953, 0.12048551120468727, 0.1409628235437037, 0.1533535523952113, 0.17288429288146012, 0.14367916142183587, 0.16412459173615215, 0.17125283792367746, 0.19335765536759994, 0.13366813752798992, 0.14119584373227775, 0.13351115626376944, 0.17156644386629896, 0.12659498332953958, 0.13399292011503305, 0.13435146572973788, 0.11938006756727136, 0.12925857673955518, 0.1205886300599333, 0.13033061693983977, 0.13563495144096024, 0.12654622870422078, 0.1314168429110934, 0.1425567903982266, 0.15836110611411422, 0.12978416854815064, 0.12053035151970357, 0.1373754138305483, 0.13349380396935356, 0.1285022885953813, 0.1250519944174613, 0.12449882861806069, 0.13666245208415667, 0.14610880977449245, 0.13220605450391124, 0.13492658461625512, 0.1100358465243825, 0.1657256801501754, 0.10741280534480944, 0.1110970280149991, 0.13378041039320937, 0.1239342684677265, 0.12233082028607985, 0.131985764824488, 0.12939283044986186, 0.13675892209587473, 0.15888455355865674, 0.12088109755891682, 0.1404232486297411, 0.12036156706541777, 0.14645199564689182, 0.13390117522205328, 0.11124807769922473, 0.10362751232598541, 0.1318548830549301, 0.11568423994842439, 0.10202448336402828, 0.12310630567908801, 0.09786583486246495, 0.1353504423774208, 0.13365969780142445, 0.12397540385261704, 0.11907012613363437, 0.13959453945093775, 0.13555388841983412, 0.11539547104031136, 0.14049860170045686, 0.1566768454386054, 0.13400035888300876, 0.14544759339431101, 0.14596941167492328, 0.11504872420860644, 0.13199479559241376,
# 0.16939786448610755, 0.1464124311206898, 0.1309601355871369, 0.1624392828183892, 0.1952673907447, 0.15726904618327195, 0.1633020066769509, 0.14229510078714463, 0.20366591223918307, 0.15413579069378597, 0.13658402047197712, 0.13216493922451814, 0.12738966246292646, 0.15290812182221922, 0.17715509765809673, 0.17563781718529797, 0.16438187757709974, 0.22719444752195614, 0.25091811540476044, 0.2484244454378663, 0.22280538057625968, 0.2415962612216638, 0.24266551508008707, 0.24566193747091078, 0.2040926233286184, 0.238563202350904, 0.1902891608758072, 0.16179768864170033, 0.1696105061463923, 0.16975576427626077, 0.18384321548018875, 0.17384866146152936, 0.1804470732096539, 0.19425367893021864, 0.18359877472672645, 0.21486526489171967, 0.2273403684678748, 0.21593841035427572, 0.22006062212375932, 0.21592450109149072, 0.215052101748005, 0.18311337075156428, 0.20818247227890674, 0.20416819178900616, 0.18989999792143755, 0.19172559361245145, 0.20145297184694638, 0.24422095833722457, 0.18443513098167721, 0.1780653707239322, 0.16716888382430384, 0.1553042182213165, 0.18632114014082893, 0.14685061825384663, 0.18209842765986914, 0.1591329633540499, 0.25500038785858364, 0.235498238606801, 0.2435775481616438, 0.17743380662122804, 0.1745662310883796, 0.1598448647959614, 0.13829918312249373, 0.1302138076436587, 0.13062293428724414, 0.15416946787198066, 0.11308512225028164, 0.13434284544209385, 0.15035105660771453, 0.13879089107689227, 0.15078647467506393, 0.14607329236904507, 0.13174574245942258, 0.1596375422729154, 0.14787504975508084, 0.15910635834476536, 0.15556826899416945, 0.15673109174742933, 0.1542269929833963, 0.15310089992748166, 0.1967084805424957, 0.2357836444055053, 0.21219830444351087, 0.20603684772139189, 0.19587322978115373, 0.2649065882861652, 0.19118259847847272, 0.16457334916500474, 0.1913889310087422, 0.16468517727601162, 0.15967965321647468, 0.19324302876523872, 0.1870816633716723, 0.1747804407394135, 0.1717667415946893, 0.1611557260666074, 0.22909555764879444, 0.17238824935208105, 0.1583178383878654, 0.1956403289213749, 0.1506111500712968, 0.16437494174639103, 0.14255191840057885, 0.12090702317157613, 0.1288509921816724, 0.1392983518315143, 0.1437884932123137, 0.1307770315366145, 0.14970742554643024, 0.13220858078093797, 0.16602945718130127, 0.16281268065750956, 0.15794461549886407, 0.16429391116927056, 0.16153900171319055, 0.13693296453996656, 0.16456021670944043, 0.1729434162892411, 0.18894597162468216, 0.18890660126116335, 0.19134125913262423, 0.16290093591400612, 0.17478051292743163, 0.16840309155121228, 0.1553487428854156, 0.15674039478828994, 0.1452148482521546, 0.152423048150132, 0.16228875772424453, 0.1677740963402655, 0.20101152252284993, 0.1281602097934211, 0.1727711979744645, 0.17410558484612912, 0.11325573180116912, 0.13428065217336171, 0.14631207215416708, 0.1493804207872604, 0.14986603039417123, 0.11868766428255124, 0.1150976148931351, 0.10490483275090856, 0.1377705812731365, 0.10120912581942633, 0.1276836183179082, 0.13067536435605795, 0.11252814251318097, 0.11291688735635699, 0.13131081398475763, 0.1354603316119504, 0.14335430273539201, 0.12517588135886346, 0.13723552982491347, 0.151961408051097, 0.1483393642264771, 0.1718272076002038, 0.15476582556730825, 0.14186718967433828, 0.1924631742503, 0.1644716295544766, 0.1677094372982274, 0.16893834446694392, 0.16991007649469006, 0.1616212390310283, 0.16632249639138066, 0.15622985063134323, 0.21820684660291206, 0.20566410890890946, 0.19647028760260135, 0.1843073741028402, 0.2263227428862037, 0.2225655777298319, 0.19252177089313147, 0.18849127538847926, 0.16503321039210223, 0.19754332152833975, 0.18743226641903304, 0.1680168866483474, 0.18738591572917684, 0.17209359424586787, 0.15740264458459297, 0.18360539965074885, 0.17681953215576032, 0.17107933586524618, 0.1725000219097839, 0.24692903615324252, 0.25792709142886583, 0.22817513908676826, 0.18641843158251198, 0.20314726272127606, 0.1891792333330171, 0.15572432549596188, 0.1258449429403741, 0.15293554970770756, 0.1561344194684245, 0.15940538115482433, 0.1420623153638629, 0.17517052529371366, 0.15102780516561615, 0.16773425198400438, 0.21955529623977477, 0.2498501041202986, 0.28801380505960245, 0.23962605284311242, 0.16470239563469277, 0.23480066397166835, 0.1811847991679833, 0.17056564921761738, 0.19394486624988275, 0.1635021346406002, 0.17663806629329085, 0.16900965306048946, 0.1893815855265883, 0.14924988983573792, 0.17159020781328135, 0.18937081454837035, 0.19459486783919347, 0.21515638971169723, 0.14608030624867818, 0.16984347253346416, 0.13475684367272628, 0.13244277549669634, 0.1472819961788559, 0.1286805027990965, 0.11596485286952839, 0.1249059388124218, 0.15924972028159154, 0.1337371941465987, 0.13539399135335467, 0.1519294651892011, 0.18011514724848796, 0.1719875881378127, 0.16181244316834692, 0.179372153734393, 0.15447182089272807, 0.19452226435751344, 0.17529078409142074, 0.15845829979044013, 0.15482986037647498, 0.19540456808655834, 0.17827949491897016, 0.18906032124415245, 0.15174529395013553, 0.14785631402156424, 0.14538502676416773, 0.11108062313872702, 0.1173454903887679, 0.1279589228983577, 0.13718103758944136, 0.17163960488179064, 0.1561886689291014, 0.12359272711925445, 0.13682167243549093, 0.1408613369633875, 0.12307306519006898, 0.13426591638478744, 0.11031172598569104, 0.13931507105961935, 0.156985374233853, 0.12437383038572543, 0.1515789993803801, 0.14727439744548285, 0.1429464119710189, 0.14760078599364887, 0.15220329599353166, 0.14757330177730602, 0.16228610840762017, 0.17343806618820481, 0.1396460733207243, 0.1501637351082041, 0.1425917836723767, 0.18471205368244115, 0.13591770361169445, 0.15377551410126392, 0.13527695179341775, 0.11163847199667565, 0.1176208958757593, 0.13760552520789146, 0.1443602918048984, 0.13750153113689445, 0.14796632242291846, 0.16336141588015268, 0.17912694108584623, 0.19684198664942157, 0.13340298546231838, 0.14282270338093084, 0.15338453911796676, 0.14982878806605113, 0.14576267813280838, 0.15389833900923997, 0.13221706431665992, 0.1599709901134757, 0.16959630725668914, 0.13412003385832277, 0.13846580633598188, 0.14525842110561316, 0.14151305508043935, 0.12988570152092074, 0.18406860241070155, 0.16846684059596234, 0.16080414339236368, 0.14638994467441713, 0.15550020001126558, 0.1773703204871909, 0.1760734926164677, 0.1459941979840474, 0.12685408278065377, 0.1740490858923048, 0.1433908861886952, 0.1532665995701928, 0.17327364111431898, 0.1575550789521626, 0.15150432359125957, 0.14964042701238628, 0.17311342624688641, 0.19814628766733788, 0.21020665149612078, 0.19044084015456106, 0.1776748955145426, 0.19213441822203747, 0.20674581238677547, 0.20350362187044035, 0.21396832980263705, 0.16081226525006992, 0.16165099547802136, 0.15167077328524592, 0.2124906740322978, 0.16265957344794155, 0.20094645234642397, 0.19534474963466075, 0.22562020659208543, 0.18194291523979533, 0.20297370905441922, 0.18497087546910176, 0.1582703792225488, 0.16283417437479947, 0.14657251281405073, 0.15911252748445495, 0.1451260177955908, 0.17934818434590258, 0.14071396170306635, 0.14624235230778546, 0.17469382343932557, 0.1441766469777554, 0.14340173299259973, 0.15822462201064602, 0.15710535621037247, 0.1408092193108301, 0.12642231757255118, 0.1814255654674125, 0.13482652532377032, 0.13329961257911926, 0.1459252517191389, 0.15148517604643308, 0.145678213024827, 0.15205628306569877, 0.16820539014602812, 0.16734657478583295, 0.14317809490167627, 0.1680631509485941, 0.1348380520034181, 0.1429367589148015, 0.09892168662484772, 0.14561998950562605, 0.151164696781012, 0.1413087713496578, 0.17230757030053698, 0.16189632301587534, 0.21772538154958612, 0.15376339309640424, 0.1309531329594751, 0.11633313351641818, 0.11771894573016288, 0.1253170118231424, 0.12538927029278946, 0.16436443255766267, 0.1344804820932579, 0.1306241363154598, 0.1465524592886083, 0.17590448675454956, 0.15899617972698904, 0.19567988520715346, 0.15380087494352304, 0.1881405138107381, 0.20438517722144614, 0.2349508728456533, 0.2041274930733613, 0.17898170403031244, 0.1847464103903079, 0.17775513484140715, 0.15822326395219233, 0.15949158296704277, 0.1909508204984491, 0.15079388303241537, 0.19360549063882923, 0.1859530558134805, 0.2271131483486665, 0.2136087455216383, 0.25603946939721056, 0.27852645398020853, 0.24169694537933178, 0.2629198468775312, 0.22790535921086097, 0.1915191117317765, 0.16857774970337855, 0.18654575584156602, 0.1409295692436273, 0.12434809420907796, 0.14125952554370094, 0.12778801865361836, 0.15428303430913043, 0.16585550484471737, 0.1508758825031965, 0.1289981904765994, 0.16282828281717526, 0.15194940772586074, 0.13250618056309996, 0.11925493273437497, 0.11734901052436177, 0.1553236938384344, 0.12498533601130753, 0.14282873369814184, 0.14337794036298412, 0.1583628358862713, 0.11353602033466086, 0.13729403714072724, 0.15825985376536494, 0.1386798724423831, 0.14465184455947372, 0.1378020150257936, 0.11904698781816236, 0.13297106904777012, 0.13894103679931197, 0.14546678384387784, 0.14125889894254506, 0.12684495710643282, 0.1484268643787052, 0.139748311441684, 0.13932387403472676, 0.11949085304574765, 0.11923376019841644, 0.11896524257585032, 0.1242748235125835, 0.12298811247350343, 0.13053120079996233, 0.15551271486765283, 0.14874319437057806, 0.14084324625941322, 0.14782405606210328, 0.12951642125811377, 0.14509794041972968, 0.14740952317045478, 0.1474312888676988, 0.0988631803812681, 0.11618291241738432, 0.11545234355966946, 0.13702973982396768, 0.11131302358089737, 0.11118004527094648, 0.1207366895573655, 0.11373432123802575, 0.1582615014380413, 0.13294237323832292, 0.11501671118087468, 0.14352699180233308, 0.12356368422548271, 0.1271701744473931, 0.10546166109386229, 0.10972450413534225, 0.11485396430658869, 0.1180954653746449, 0.13265854364926796, 0.12271538625636873, 0.14544346936647726,
# 0.1287166338967618, 0.12824528996529122, 0.12900225055583975, 0.14374736917692238, 0.131699821095117, 0.13160867670221957, 0.11658198997276426, 0.13030669234955536, 0.11533892946419745, 0.1268203467306404, 0.13372573932786516, 0.12515933884957028, 0.11939282365984047, 0.13603314891635987, 0.1364763625491538, 0.11510182708659422, 0.11791623310955966, 0.12534511602756304, 0.12326424540734826, 0.18125132452292075, 0.14201106437939656, 0.17019966351556073, 0.14117495838612262, 0.11733825841782562, 0.16507894253614205, 0.16633119593829296, 0.16077401785321394, 0.13975785238350572, 0.1449899993206494, 0.17156761390393796, 0.159924841306859, 0.13215410724579293, 0.19265977815167648, 0.15183431011192672, 0.15592766455643847, 0.17724890365738175, 0.19234465570888956, 0.20917106861975107, 0.28322223964539267, 0.22305168498257996, 0.27319879479596004, 0.2508778817501682, 0.25063378808465336, 0.22204771292423633, 0.19205456631561457, 0.2163598503248641, 0.20275975663128248, 0.1854273130367317, 0.1720827953008369, 0.2010170545753141, 0.1417536024082096, 0.21450559129340663, 0.17489130772388575, 0.19842331200587787, 0.18850756217870618, 0.23020689813656364, 0.23041172264173446, 0.2389337588271404, 0.20660377847037395, 0.21736879241902043, 0.21023275764321347, 0.1996108734291573, 0.195545034750681, 0.21014440304634124, 0.23472783269954325, 0.18926972646743376, 0.228795606061]

# def hess_maker(pdbfile, pdbcode, Temp, cutoff, gtype='b'):
#     BU = BfactorSolver(pdbfile, pdbcode, Temp, cutoff=cutoff, load_inv_hess=False)
#     BU.compare_ana_vs_exp(str(pdbcode)+ str(cutoff) + '.png', gtype)
#     BU.save_inverse_Hessian(str(pdbcode) + '_ib_H_' + str(cutoff))

# pdbfile = '6vxx.pdb'
# pdbcode = '6vxx'
# Temp = 298
# cs = [x for x in range(25, 35) if x % 5 ==0]
# print(cs)
# comms = [(pdbcode, pdbfile, Temp, x) for x in cs]
# print(comms)
# for x in comms:
#     pc, pf, t, c = x
#     mp.Process(target=hess_maker, args=(pc, pf, t, c)).start()


# BU.load_inverse_Hessian('ib_H_25.npy')
# BU.calc_rawmsds()
# print(len(BU.rawmsds))
# print(len(BU.exp_bfactors))
# bi = [0, 3, 69, 70, 322, 337, 338, 339, 340, 341, 349, 350, 351, 352, 353, 354, 355, 466, 467, 468, 469, 472, 476, 478, 498, 499, 500, 501, 502, 503, 505, 506, 507, 508, 509, 510, 511, 565, 566, 567, 568, 569, 570, 571, 572, 581, 634, 635, 636, 637, 769, 770, 771, 772, 773, 785, 819, 820, 831, 834, 835, 836, 837, 838, 839, 840, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 941, 942, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 999, 1002, 1003, 1004, 1005, 1061, 1062, 1130, 1131, 1312, 1313, 1314, 1325, 1326, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485]
# BU.fit_to_exp()

# BU.load_sim_rmsds(flu125)
# # BU.compare_all('no_extrap.png', 'b')
# BU.extrapolate_exp_msd(300)
# BU.extrapolate_ana_msd(300)
# BU.compare_all('extrap.png', 'b')
# o=open('exp_bfactors.txt', 'w')
# lbled = list(zip(np.arange(0,len(BU.exp_bfactors),1),BU.exp_bfactors))
# print(lbled, file=o)
# o.close()
# BU.save_inverse_Hessian('ib_H_20')
# print(BU.ana_gamma)
# print(BU.sim_gamma)

# BU = BfactorSolver('ca', 'ca.pdb', 110, load_inv_hess=False)
# baindxs = [0,1,2,3]
# BU.fit_to_exp_nopeaks(baindxs)
# BU.compare_ana_vs_exp('catrial.png', 'b', customdomain=[3, BU.cc])

# bbindxs = list(np.arange(40,120,1))
# CB = BfactorSolver('cb', 'cb.pdb', 110, load_inv_hess=False)
# CB.fit_to_exp_nopeaks(bbindxs)
# CB.compare_ana_vs_exp('cbtrial.png', 'b')

# BFACTOR CHECK
# g1 = BfactorSolver('sgfp', 'sgfp.pdb', 300, solve=True)
# g1.load_sim_rmsds_from_file('priordevs.json')
# # x1 =  g1.sim_msds
# g1.load_sim_rmsds_from_file('devsgfp4.954.json')
# g1.compare_all('gfptrial.png', 'b')
# x2 = g1.sim_msds


###START HERE FOR SPIKE PROTEIN
# S = Structure_Maker('6vxx', './6vxx.pdb', 300, ptype='monomer', cutoff=15)

# S = Structure_Maker('histone', './histone.pdb', 300, ptype='monomer', cutoff=15)
# g1 = S.Solvers[0]
# g1.load_inverse_Hessian('6vxx.pdb_ib_H_15.npy')
# # g1.load_sim_rmsds_from_file('./1bu4/dbu.json')
# # g1.load_sim_rmsds_from_file('./devssq7.json')
# # g1.fit_to_exp_nopeaks([0, 1,2,3,4,5,6,7,8,9,10])
# # g1.compare_all('spikefit.png', 'b')
# bids = [372, 1344, 2316]
# g1.calc_rawmsds()
# g1.fit_to_exp_nopeaks(bids)
# g1.calc_bonds()
# for x, y, d in g1.bonds:
#     if x == 372 or y == 372:
#         print(x, y, d)
# g1.compare_ana_vs_exp('histfit.png', 'b')


# # free_compare('xvcheck.png', x1, x2, legends=['prior','post'])

# bt = BfactorSolver ('j', [0, 0, 0], 300, [0,0,0], load_inv_hess=False, solve=False, load_test_sys=True)


#Implemented this paper's idea https://pubs-rsc-org.ezproxy1.lib.asu.edu/en/content/articlepdf/2018/cp/c7cp07177a
#Very Basic Implementation for C-A coarse graining
# class MVPANM(ANM):
#     def __init__(self, coord, exp_bfactors, cutoff=15, scale_resolution=15, k_factor=3, algorithim='ge', T=300):
#         super().__init__(coord, exp_bfactors, T=T, cutoff=cutoff)
#         # weight factor might change later we'll see
#         self.w = 1.
#         self.scale_resolution = scale_resolution
#         self.k_factor = k_factor
#         self.alg = algorithim
#         # self.cutoff = cutoff
#
#         self.spring_constant_matrix = []
#         # Rigidity Functions, only calculate once and unique to each system
#         self.kernels = []
#         self.mu = []
#         self.mu_s = []
#         self.model_id = 'MVP'
#
#     def algorithim(self, dist):
#         # Can choose between Generalized Exponential and Generalized Lorentz Function
#         def gen_exp(dist):
#             return math.exp((-1. * dist / self.scale_resolution)) ** self.k_factor
#
#         def gen_lor(dist):
#             return 1. / (1. + (dist / self.scale_resolution) ** self.k_factor)
#
#         if self.alg == 'ge':
#             return gen_exp(dist)
#         elif self.alg == 'gl':
#             return gen_lor(dist)
#
#     def mvp_compute_all_rigidity_functions(self):
#         self.kernels = []
#         self.mu = []
#         self.mu_s = []
#         for i in range(self.cc):
#             ker_i = 0.
#             for j in range(self.cc):
#                 d = dist(self.coord, i, j)
#                 if self.cutoff > 0. and d <= self.cutoff:
#                     ker = self.algorithim(d)
#                 elif self.cutoff > 0. and d > self.cutoff:
#                     ker = 0.
#                 else:
#                     ker = self.algorithim(d)
#                 self.kernels.append(ker)
#                 ker_i += ker * self.w
#             self.mu.append(ker_i)
#
#         # replace ii with sum
#         for i in range(self.cc):
#             indx = i * self.cc + i
#             self.kernels[indx] = -1 * self.mu[i]
#
#         # Normalized density funciton
#         mu_s = []
#         min_mu = min(self.mu)
#         max_mu = max(self.mu)
#         for i in range(self.cc):
#             mu_normed = (self.mu[i] - min_mu) / (max_mu - min_mu)
#             mu_s.append(mu_normed)
#         self.mu_s = mu_s
#
#     def mvp_compute_gamma_1(self, i, j):
#         return (1. + self.mu_s[i]) * (1. + self.mu_s[j])
#
#     def mvp_compute_gamma_2(self, i, j):
#         indx = i * self.cc + j
#         return self.kernels[indx]
#
#     def mvp_compute_spring_constants(self):
#         if self.kernels and self.mu and self.mu_s:
#             sc_matrix = np.full((self.cc, self.cc), 0.0)
#             for i in range(self.cc):
#                 for j in range(self.cc):
#                     if i == j:
#                         spring_constant_ij = 1.
#                     else:
#                         spring_constant_ij = self.mvp_compute_gamma_1(i, j) * self.mvp_compute_gamma_2(i, j)
#                     sc_matrix[i, j] = spring_constant_ij
#             self.spring_constant_matrix = sc_matrix
#         else:
#             print('Must Compute Rigidity Functions Prior to Spring Constants')
#
#     def simplify_matrix(self, percentile):
#         cut_val = np.percentile(self.spring_constant_matrix, percentile)
#         for i in range(self.cc):
#             for j in range(self.cc):
#                 if self.spring_constant_matrix[i, j] < cut_val:
#                     self.spring_constant_matrix[i, j] = 0
#
#     def mvp_calc_bfactors(self, cuda=False):
#         self.calc_dist_matrix()
#         hess = self.calc_hess_fast_sc(self.spring_constant_matrix)
#         iH = self.calc_inv_Hess(hess, cuda=cuda)
#         self.calc_msds(iH)
#         self.ana_bfactors = [self.bconv * x for x in self.msds]
#
#     def mvp_fit_to_exp(self):
#         try:
#             from sklearn.linear_model import LinearRegression
#         except ImportError:
#             print('Check that sklearn module is installed')
#             sys.exit()
#         print(self.ana_bfactors)
#         flex_data = np.asarray(self.ana_bfactors)
#         exp_data = np.asarray(self.exp_bfactors)
#         X = flex_data.reshape(-1, 1)
#         Y = exp_data
#         print(flex_data)
#         fitting = LinearRegression(fit_intercept=False)
#         fitting.fit(X, Y)
#         slope = fitting.coef_
#         self.spring_constant_matrix /= slope
#         self.ana_bfactors *= slope
#
#     def mvp_theor_bfactors(self, outfile):
#         free_compare(outfile, self.exp_bfactors, self.ana_bfactors,
#                          legends=['Experimental  (PDB)', 'Analytical (MVP)'])
#
#     def calc_mvp(self, cuda=False):
#         self.mvp_compute_all_rigidity_functions()
#         self.mvp_compute_spring_constants()
#         self.mvp_calc_bfactors(cuda=cuda)
#         print(self.spring_constant_matrix)
#         self.mvp_fit_to_exp()