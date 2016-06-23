#!python2.7
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:48:20 2016
Author: cpashartis

This code was written by Christopher Pashartis, cpashartis@gmail.com

Tight Binding LCAO approach
"""

import numpy as np
import pymatgen as pm
from sys import exit
from re import findall
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh

class LCAO:
    
    def __init__(self, filename):
        
        """Method for determining the structure of the file being input,
        currently limited to crystal / wigner - seitz structures. Import the
        cif file here for use of atomic structures"""
        
        self.struct = pm.Structure.from_file(filename, primitive=False)   
        
    @staticmethod
    def diamond_g(d, k):
        """Diamond structure plane wave components, 
        returns vector for all k points"""        
        return_array = \
             1/4.*np.array([np.exp(1.j*np.dot(d[0,:], k)) \
                + np.exp(1.j*np.dot(d[1,:], k))\
                + np.exp(1.j*np.dot(d[2,:], k)) + np.exp(1.j*np.dot(d[3,:], k)),
               np.exp(1.j*np.dot(d[0,:], k)) + np.exp(1.j*np.dot(d[1,:], k))\
               - np.exp(1.j*np.dot(d[2,:], k)) - np.exp(1.j*np.dot(d[3,:], k)),
               np.exp(1.j*np.dot(d[0,:], k)) - np.exp(1.j*np.dot(d[1,:], k))\
               + np.exp(1.j*np.dot(d[2,:], k)) - np.exp(1.j*np.dot(d[3,:], k)),
               np.exp(1.j*np.dot(d[0,:], k)) - np.exp(1.j*np.dot(d[1,:], k))\
               - np.exp(1.j*np.dot(d[2,:], k)) + np.exp(1.j*np.dot(d[3,:], k))])

        return return_array
       
    @staticmethod
    def diamond_nn_H(g, (delta_E, V_ss, V_sp, V_xx, V_xy), sym_fac ):
        
        """Diamond tight binding interaction energy term for nearest neighbour
        interactions of s and p type orbitals 4x4 matrix from Peter Young
        semiconductor physics. Where nearest neighbours are required for each
        element"""
        
        E_p = 0
        E_s = -1.*delta_E
        
        V_sp = V_sp * sym_fac
        V_xy = V_xy * sym_fac
        
        H = np.matrix([[E_s, V_ss * g[0], 0,0,0,V_sp*g[1], V_sp*g[2], V_sp*g[3]],
        [V_ss*np.conj(g[0]),E_s, -1*V_sp*np.conj(g[1]), -1*V_sp*np.conj(g[2]),\
        -1*V_sp*np.conj(g[3]),0,0,0],
        [0,-1*V_sp*g[1], E_p, 0,0,V_xx*g[0], V_xy*g[3],V_xy*g[2]],
        [0,-1*V_sp*g[2], 0, E_p, 0, V_xy*g[3], V_xx*g[0], V_xy * g[1]],
        [0, -1*V_sp*g[3], 0, 0 , E_p, V_xy*g[2], V_xy*g[1], V_xx*g[0]],
        [V_sp*np.conj(g[1]), 0, V_xx*np.conj(g[0]), V_xy*np.conj(g[3]),\
         V_xy*np.conj(g[2]), E_p, 0,0],
        [V_sp*np.conj(g[2]), 0, V_xy*np.conj(g[3]), V_xx*np.conj(g[0]),\
        V_xy*np.conj(g[1]), 0, E_p,0],
        [V_sp*np.conj(g[3]), 0, V_xy*np.conj(g[2]), V_xy*np.conj(g[1]),\
        V_xx*np.conj(g[0]), 0, 0, E_p]])
       
        return H
       
    def k_points(self, sym_pts, coarse = 0, struct_t = 'diamond'): 
                    
        #assuming array
        sym_pts = np.matrix(sym_pts)*np.matrix(self.struct.reciprocal_lattice.matrix)
        if coarse == 0 :
            coarse = np.array([50,50,25,25])
            
        if len(coarse) != len(sym_pts) - 1:
            raise IndexError("The length of the coarse \
                should be one less sym_points")
        
        sym_pts = np.array(sym_pts) #convert back for np useage
        start_pt = sym_pts[0,:]
        kpts = []
        #generate grid
        for ind in range(1,sym_pts.shape[0]):
            end_pt = sym_pts[ind,:]
            #double counts endpts
            kx,ky,kz = [np.linspace(start_pt[i], end_pt[i], coarse[ind-1]).reshape(coarse[ind-1],1) for i in range(3)]
            kpts.extend(np.concatenate((kx,ky,kz), axis = 1))
            start_pt = end_pt.copy()
        return kpts
       
    def find_neighbours(self):
        """Generate nearest neighbours from pymatgen code to prep for hamiltonian"""
        #for each atom site find nearest neigbours
        #assume diamond structure
            
        #right about here we run into a problem, we need to find the
        #most primitive cell in order to avoid using lattice constants as well
        #as to maintain convention that the brillouin zone kpts are defined
        #over the primitive cell
        
        #so here is what we are going to do, replace all atoms with the same
        #and reduce to find prim_struct
        elements = self.struct.composition.elements
        prim_struct = self.struct.get_primitive_structure()
        #replace all with first element
        for ele in elements[1:]:
            prim_struct.replace_species({ele:elements[0]})
        #now find prim structure
        prim_struct = prim_struct.get_primitive_structure()
        
        try:
            dist_mat = prim_struct.distance_matrix
            tol = 1E-5
            #dealing with issue of 1E-15 in distance
            dist_mat[dist_mat < tol] = 0.0 #set anything less than tol to -
            nn_distance = min(dist_mat[np.nonzero(dist_mat)])
        except ValueError: #check if prim_struct has only one atom
            if len(prim_struct.sites) == 1 :
                neighbors = 'fix'
            else:
                print "I have no idea what to do duuuude"
                exit()
        
        neighbors = self.struct.get_all_neighbors(nn_distance+0.01) #0.01 to get all
                
        return neighbors, prim_struct
        
    @staticmethod
    def load_tb_coefs(atom, adj_atoms, bond_length = 1, bond_type = 'sp3',
                      sub_type = 'orb', dist_scale = '2', **kwargs):
                          
        """Pull tight binding parameters from Parameters file. Only capable
        of binary and sp3 alloys at the moment. It will average S and P E orbital
        energies to accomodate changing values in meV
        
        ex. Ga surrounded by 3 As and 1 Bi then the E_s will be 3 parts GaAs
        
        --------
        atom - atom name string
        adj_atoms - list of adjacent atoms, it will check if single atom crystal
        sub_type - can be 'orb' or 'bond'
        bond_type - sp3, or sp3s (last s is excited)
        dist_scale - 2 for the square law, may implement from file later
        
        returns an array of orbital energies or bond energies in the format of 
        files III-V_binaries for binaries or single_atom for single atom data
        """
        
        file_name = ''
        if bond_type not in ['sp3','sp3s']:
            print "cannot do this yet"
            exit()
        
        if type(adj_atoms) is not list:
            adj_atoms = [str(adj_atoms)]
            
        if adj_atoms.count(atom) == len(adj_atoms):
            file_name = 'single_atom.dat'
        else:
            #get filename from keywargs
            for key,arg in kwargs.iteritems():
                if key == 'file_name':
                    file_name = arg
                    
            if bond_type == 'sp3':                    
                if file_name == '':
                    file_name = 'III-V_binaries_sp3.dat'
                    
                num_orb = 2
            elif bond_type == 'sp3s':
                if file_name == '':
                    file_name = 'III-V_binaries_sp3s*.dat'
                    
                num_orb = 3 
                
        print "Using data from %s" %(file_name), kwargs
        #-----------------
        #DEV note, handles sp3s and sp3 in the same fashion, just accounts
        #for two extra indices in one case
        #-----------------
        if sub_type == 'orb':
            E = np.zeros(2*num_orb)
            stop_cntr = 0
            with open('Parameters/' + file_name, 'r') as tb_in:
                for line in tb_in:
                    #check for comments
                    if line[0] =='#':
                        continue
                    line = line.strip()
                    line = line.split()
                    if atom in line[0]:
                        if 'single' in file_name:
                            return np.array([0,line[1]], dtype = float), line[0]
                        else:
                            #we have a match, find if cation or anion
                            break_alloy = findall(r'[A-Z][a-z]*', line[0])
                            cat_an_i = break_alloy.index(atom)
                        #check if other atom in neigh
                        if break_alloy[cat_an_i-1] in adj_atoms:
                            E += np.array(line[2: 2+2*num_orb],
                                dtype = float)
                                
                            #add the number of times it has the same neighbor
                            E *= adj_atoms.count(break_alloy[cat_an_i-1])
                            stop_cntr += adj_atoms.count(break_alloy[cat_an_i-1])
                        
                        if stop_cntr == 4 :
                            #stop and return the average orbital energy
                            return E/(4.0), line[0]
                            
                print "well something went wrong and I couldn't find the data"
                
        elif sub_type == 'bond':
            with open('Parameters/'+file_name, 'r') as tb_in:
                for line in tb_in:
                    #check for comments
                    if line[0] =='#':
                        continue
                    line = line.strip()
                    line = line.split()
                    if atom in line[0]:
                        #no bond scaling here
                        if 'single' in file_name:
                            return np.array(line[2:], dtype = float), line[0]
                        else:
                            #we have a match, find if cation or anion
                            break_alloy = findall(r'[A-Z][a-z]*', line[0])
                            cat_an_i = break_alloy.index(atom)
                        #check if other atom in niegh
                        if break_alloy[cat_an_i-1] in adj_atoms:
                            if dist_scale == '2': 
                                return np.array(line[num_orb*2+2:], dtype = float)\
                                *((float(line[1])/bond_length)**2), line[0]
                            else:
                                print "cannot do bond scaling other than 2 for now"
                                exit()
                            
        
    def build_diamond_nn_H(self, g, k_points, H_type = 'sp3', **kwargs):#, int_function = 0):
        """Assumes use of crystal structure
        
        The input must be dictioniaries of the atom in which the overlaps
        or energies are describing, ex.
        
        Args:
            params - (delta_E, V_ss, V_sp, V_xx, V_xy), is a tuple of the
                    tightbinding coefficients, if only sigma and pi bonds known
                    see method diamond_init
            dict_atom_tight - dictionary with element symbol as key and with 
                            delta_E, E_xx
            H_type  - sp3, sp3s, (maybe sp3d5s later)
                            
        """
        
        if H_type == 'sp3':
            interactions = 4
        elif H_type == 'sp3s':
            interactions = 5

        neighbors, prim_struct = self.find_neighbours()
        eig_list = []
        eig_val_l = []
        eig_vec_l = []
        atom_tb_coefs = {}
        alloy_tb_coefs = {}
        
        size = len(self.struct.sites) * interactions
        for kpt in k_points:
            #we want to build a (4xN)x(4xN) matrix
            
            #loop through each site's neighbors
            site_i = 0
            H = np.matrix(np.zeros((size,size)), dtype = complex)
            
            for nearests in neighbors:

                #find atom name and pull tb params from data
                atom_name = str(self.struct.sites[site_i].specie.symbol)
                neigh_name = [str(x[0].specie.symbol) for x in nearests]
                #atom name first, followed by neighbor
                temp = [atom_name]
                temp = neigh_name[:]
                temp.append(atom_name)
                temp.sort()
                temp2 = []
                #reducing number of permutations, GaAsAsAsAs is AsGa
                for ele in temp:
                    if ele not in temp2:
                        temp2.append(ele)
                temp = temp2[:]
                atom_key = ''.join(temp)
                #remove duplicates from the atom_key
                atom_key = atom_key.strip()
                if atom_key not in atom_tb_coefs.keys():
                    data,place = self.load_tb_coefs(atom_name, neigh_name,
                                bond_type = H_type, sub_type = 'orb', **kwargs)
                    atom_tb_coefs.update({atom_key:(data,place)})
                    
                data, place = atom_tb_coefs[atom_key]
                #ord_alloy_name is one of the names of the bonds, meant for placeholder
                #of anion or cation
                #find if the atom is anion or cation based on position in string
                break_alloy = findall(r'[A-Z][a-z]*', place)
                if len(break_alloy) == 1: 
                    cat_an = -1
                    E_orb = data
                else:
                    cat_an = break_alloy.index(atom_name) #0 for cation
                    E_orb = np.array([data[j] for j in range(1-cat_an,\
                        len(data),2)])
#------------------------------------------------------
#Generate Hamiltonian for a specific kpt.
                
                #generate hamiltonian for this specific element (i.e. 4x4 matrix)
                
    #---------
    #first build diagonal i.e. s1,px1, py1, pz1  s1, px1, py1, pz1
                if H_type == 'sp3':
                    H_diag = np.diag([E_orb[0], E_orb[1],E_orb[1],E_orb[1]])
                elif H_type == 'sp3s':
                    H_diag = np.diag([E_orb[0], E_orb[1],E_orb[1],E_orb[1],\
                            E_orb[2]])
                            
                H[interactions*site_i:interactions*(site_i+1),
                  interactions*site_i:interactions*(site_i+1)]= H_diag   
                  
                for nearest in nearests:
                    
                    bond_length = nearest[1]
                    nearest = nearest[0] #it was a tuple
                    d = nearest.coords - self.struct.sites[site_i].coords
    #---------
    #build off diagonals
                    #first find if one of the neighbors is in the structure
                    #check periodic\
                    site_off_i = -1
                    for i in self.struct.sites:
                        if nearest.is_periodic_image(i):
                            site_off_i = self.struct.sites.index(i)
                            #break
                    #if no periodic image then we don't use in hamiltonian
                    if site_off_i == -1:
                        continue
                    
    #---------
    #pull existing tb coefficients
                    neigh_name = str(nearest.specie.symbol)
                    temp = [atom_name, neigh_name]
                    temp.sort()
                    alloy_name = ''.join(temp) #to keep permutations down
                    if alloy_name not in alloy_tb_coefs.keys():
                        data, act_key = self.load_tb_coefs(atom_name, neigh_name,
                                        bond_type = H_type,sub_type = 'bond',
                                        bond_length = bond_length, **kwargs )
                        alloy_tb_coefs.update({alloy_name:(data, act_key)})
                       
                    #load data
                    data,ord_alloy_name = alloy_tb_coefs[alloy_name]
                    
                    
                    #unpack data
                    #sepSig is excited s with p sigma bond
                    #cation case assume
                    #cat_an carried from outside loop
                    if cat_an == -1 :
                        V_sSig, V_spSig0, V_ppSig, VppPi = data
                        V_spSig1 = V_spSig0
                    else: #binary
                        if H_type == 'sp3':
                            V_sSig, V_spSig1, V_spSig0, V_ppSig, VppPi = data
                        elif H_type == 'sp3s':
                            V_sSig, V_spSig1, V_spSig0, V_ppSig, VppPi, V_sepSig1,\
                                V_sepSig0 = data
                            
                    #this is required since we must account for anion s orbital
                        if cat_an == 1: #it is the anion, so reverse order
                            #flip Sig order
                            temp = V_spSig1
                            V_spSig1 = V_spSig0
                            V_spSig0 = temp
                            if H_type == 'sp3s':
                                temp = V_sepSig1
                                V_sepSig1 = V_sepSig0
                                V_sepSig0 = temp
                      
                    #now add components
                    phase = np.exp(1.0j*np.dot(d, kpt))
                    l = np.dot(d/np.linalg.norm(d),[1,0,0])
                    m = np.dot(d/np.linalg.norm(d),[0,1,0])
                    n = np.dot(d/np.linalg.norm(d),[0,0,1])
                    #ss | s1 p_x2  s1 p_y2 | s1 p_z2
                    #px1 s2 -1 for diff direc | px1 px2 | px1 py2 | px1 pz2
                    #py1 s2 (-1) | py1 px2 | py1 py2 | py1 pz2
                    #pz1 s2 (-1) | pz1 px2 | pz1 py2 | pz1 pz2
                    
                    #VspSig0 is the s orbital (of the atom) to the neigh p
                    #VspSig1 is the s orbital (of neigh) to atom p
                    H_off = np.matrix([
    [V_sSig * phase, V_spSig0 * l * phase,
         V_spSig0 * m * phase, V_spSig0 * n * phase], 
    [V_spSig1 *-1* l * phase, (V_ppSig * l**2 + V_ppPi *(1-l**2))*phase,
         l*m*(V_ppSig-V_ppPi)*phase, l*n*(V_ppSig-V_ppPi)*phase],
    [V_spSig1 *-1* m * phase, m*l*(V_ppSig-V_ppPi)*phase,
         (V_ppSig * m**2 + V_ppPi *(1-m**2))*phase,
        m*n*(V_ppSig-V_ppPi)*phase],
    [V_spSig1 *-1* n * phase, n*l*(V_ppSig-V_ppPi)*phase, 
         n*m*(V_ppSig-V_ppPi)*phase,
        (V_ppSig * n**2 + V_ppPi *(1-n**2))*phase] ] )
                                 
                    #stack with sp3s if sp3s
                    #note the sp3s type binding doesnt include s* s* int.
                    #only p and s*, so 3 more interactions
                    if H_type == 'sp3s':
                        H_off = np.hstack((H_off, np.zeros((4,1)) ))
                        H_off = np.vstack((H_off, np.zeros((1,5)) ))
                        #this is the first row from H_off just for s* now
                        H_off[-1,1:-1] = np.matrix([\
            V_sepSig0 * l * phase,V_sepSig0 * m * phase, V_sepSig0 * n * phase])
                        #like first column from H_off
                        H_off[1:-1,-1] = np.matrix([
                        [V_sepSig1 *-1* l * phase],
                        [V_sepSig1 *-1* m * phase],
                        [V_sepSig1 *-1* n * phase]])
                        
                    H[interactions*site_i:interactions*(site_i+1),
            interactions*site_off_i:interactions*(site_off_i+1)] += H_off
                
    #---------
                site_i += 1
#Generate Hamiltonian for a specific kpt.
#------------------------------------------------------ 
            
            #tolerance for checking if hermitian
            if not (H - H.H <= np.ones(H.shape)*1E-12).all():
                print "Matrix is not hermitian, something went wrong..."
                exit()
                
            #find eigenvalues for each kpt
            H_sp = csc_matrix(H)
            eig_list.append(np.linalg.eigvalsh(H))
            temp = eigsh(H_sp)
            eig_val_l.append(temp[0])
            eig_vec_l.append(temp[0])
            
            
        eig_val_l = np.array(eig_val_l)
        eig_list = np.array(eig_list)
        
        return eig_list, k_points
    
    @staticmethod
    def plot_eigen(eig_values):
        """Needs to be fixed for genericnouss"""
        
        from matplotlib import pyplot as plt
        from matplotlib import rc_file
        rc_file('/Users/cpashartis/bin/rcplots/paper_multi.rc')
        
        f = plt.figure()
        ax = f.add_subplot(111)
        x = np.linspace(0,1,eig.shape[0])
        for i in range(eig.shape[1]):
            ax.plot(x, eig[:,i])
        labels = ['L', r'$\Gamma$', 'X', 'K', r'$\Gamma$']
        #fix for consistancy
        plt.xticks([0, 1/3., 2/3.,5/6., 1], labels)
        #ax.set_ylim(-1, 6)
        lims = plt.ylim()
        plt.vlines([1/3.,2/3.,5/6.],lims[0],lims[1], color = 'k', linestyle = '--')
        ax.set_ylabel('Energy (eV)')
        f.savefig('band_dispersion.pdf', dpi = 1000 )
        plt.close('all')
        
             
        
if __name__ == '__main__':
    
    E_s = 0
    E_p = 7.2
    V_ssig = -2.032
    V_sp = 2.546 #np.sqrt(3)/4*5.8
    V_ppSig = 4.5475
    V_ppPi = -1.085

    file_name = '/Users/cpashartis/Box Sync/Masters/LCAO_tightbinding/test_cifs/GaAs_2atom_wann.cif'
    struct_t = "diamond"                               
    #zinc-blende / diamond
    sym_pts = np.array([[0.5, 0.5,0.5],[0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.5], [5/8.,1/4.,5/8.], [0,0,0]],
                                    dtype = float)
    a = LCAO(file_name)
    eig, kpt = a.build_diamond_nn_H(a.diamond_g, a.k_points(\
        sym_pts = sym_pts), H_type = 'sp3s')#, file_name = 'III-V_wannier_sp3.dat')
    LCAO.plot_eigen(eig)