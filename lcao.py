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
from matplotlib import pyplot as plt
from matplotlib import rc_file
rc_file('/Users/cpashartis/bin/rcplots/paper_multi.rc')

class LCAO:
    
    def __init__(self,filename):
        
        """Method for determining the structure of the file being input,
        currently limited to crystal / wigner - seitz structures. Import the
        cif file here for use of atomic structures"""
        
        self.struct = pm.Structure.from_file(filename, primitive = False)   
        
    @staticmethod
    def diamond_init(delta_E, V_ssSigma, V_spSigma, V_ppSigma, V_ppPi):
        
#        V_ss = 4. * V_ssSigma
#        V_sp = 4. * V_spSigma/np.sqrt(3)
#        V_xx = (4.*V_ppSigma/3) + (8.*V_ppPi/3)
#        V_xy = (4.*V_ppSigma/3) - (4.*V_ppPi/3)
        
        return (delta_E, V_ss, V_sp, V_xx, V_xy)
        
    #def _func_exp(self, d, k):
        
    @staticmethod
    def diamond_g(d, k):
        """Diamond structure plane wave components, 
        returns vector for all k points"""
        #may be forgetting pi here
        #make sure g is complex
        
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
       
    @staticmethod
    def k_points(sym_pts = 0, coarse = 0): 
        
        if sym_pts == 0:
            sym_pts = np.array([[.5,.5,.5], [0,0,0], [1,0,0]], dtype = float)
#            L,GAMMA,X,GAMMA
#            sym_pts = np.array([[0.5, 0.5, 0.5],[0.0, 0.0, 0.0],
#                                [0.0, 0.5, 0.5]],
#                                dtype = float)
        if coarse == 0 :
            coarse = np.array([100,100])
            
        if len(coarse) != len(sym_pts) - 1:
            raise IndexError("The length of the coarse \
                should be one less sym_points")
        
        start_pt = sym_pts[0,:]
        kpts = []
        #generate grid
        for ind in range(1,len(sym_pts)):
            end_pt = sym_pts[ind,:]
            diff = end_pt-start_pt
            #double counts endpts
            kx,ky,kz = [np.linspace(start_pt[i], end_pt[i], coarse[ind-1]).reshape(coarse[ind-1],1) for i in range(3)]
            kpts.extend(np.concatenate((kx,ky,kz), axis = 1))
            start_pt = end_pt.copy()
        #print kpts
        #kpts = [[0.,0.,0.]]
        return np.array(kpts)*2*np.pi#self.struct.lattice.a #assuming similar dimensional lattice
       
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
                #nn_distance = prim_struct.lattice.a/
                neighbors = 'fix'
            else:
                print "I have no idea what to do duuuude"
                exit()
        
        neighbors = self.struct.get_all_neighbors(nn_distance+0.01) #0.01 to get all
                
        return neighbors, prim_struct
          
#    def _off_diag_diamond(self, sym_fac):
        
        
    def build_diamond_nn_H(self, g, k_points, params , interactions = 4):#, int_function = 0):
        """Assumes use of crystal structure
        
        The input must be dictioniaries of the atom in which the overlaps
        or energies are describing, ex.
        
        Note this only works if the structure given has the typical lattice
        vectors (or factors of these vectors)
        
        0.5  0.0  0.5
        0.0  0.5  0.5
        0.5  0.5  0.0
        
        Currently assumes input of sublattice order.
        
        Args:
            params - (delta_E, V_ss, V_sp, V_xx, V_xy), is a tuple of the
                    tightbinding coefficients, if only sigma and pi bonds known
                    see method diamond_init
            dict_atom_tight - dictionary with element symbol as key and with 
                            delta_E, E_xx
                            
        CAN MAKE FASTER BY GETTING RID OF G AND JUST BUILDING PLACEMENT
                            
        """

        #print params
        delta_E, V_ss, V_sp, V_xx, V_xy = params
        neighbors, prim_struct = self.find_neighbours()
        eig_list = []
#        #sym_neg = {}
#        sym_neg = {str(prim_struct.sites[0].frac_coords):1} #assign first as 1
#        sym_counter = -1 #start at negative for positive starting site
        
        #get the unitless distance vector
        a0 = np.sqrt(2)* self.struct.lattice.a
        neigh_dist_frac = self.struct.sites[0].coords[0] - \
            neighbors[0][0][0].coords[0]/a0
        d = np.array([  [1, 1, 1],
                        [1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1] ]) *1/4.#* neigh_dist_frac
        lat_fac = 1
        size = len(self.struct.sites) * interactions
        for kpt in k_points:
            #we want to build a (4xN)x(4xN) matrix
            ################################################
            #loop to find path of -1, +1s, by starting with one site and adding
            #neighbors and keep repeating
            #this could be fixed (somehow) by determining sublattices
            #NNNNNNNNNEEEEEEEEEEDED STILL
            ################################################################
            
            #loop through each site's neighbors
            site_i = 0
            H = np.matrix(np.zeros((size,size)), dtype = complex)
            cur_g = g(d,kpt)
            for nearests in neighbors:
                #sym_counter *=-1 #multiply by negative 1 for each new position
                #format is list of [sites]
                    
                #swap if element is halfway through, i.e. next sublattice
                if len(self.struct.sites)/2 - 1 == site_i:
                    lat_fac = -1
#------------------------------------------------------
#Generate Hamiltonian for a specific kpt.
                
                #generate hamiltonian for this specific element (i.e. 4x4 matrix)
                
    #---------
    #first build diagonal i.e. s1,px1, py1, pz1  s1, px1, py1, pz1
                H_diag = np.matrix([ [E_s, 0, 0, 0],
                                     [0, E_p, 0, 0],
                                     [0, 0, E_p, 0],
                                     [0, 0, 0 , E_p] ])
                H[interactions*site_i:interactions*(site_i+1),
                  interactions*site_i:interactions*(site_i+1)]= H_diag     
  
                #doesn't work DUE TO PICKING DIFFERENT ORDER OF R FOR G .... >.<
                for nearest in nearests:
                    
                    nearest = nearest[0] #it was a tuple
#                    neighb_coord = nearest.coords - self.struct.sites[i].coords
#                    #get rid of lattice constant as the diamond k points are unitless
#                    a0 = np.sqrt(2)* self.struct.lattice.a
#                    d.append(neighb_coord/a0)
    #---------
    #build off diagonals
                    #first find if one of the neighbors is in the structure
                    #check periodic
                    #print nearest
                    site_off_i = -1
                    for i in self.struct.sites:
                        if nearest.is_periodic_image(i):
                            site_off_i = self.struct.sites.index(i)
                            #break
                    #if no periodic image then we don't use in hamiltonian
                    if site_off_i == -1:
                        continue
                            
                    #so if it is true, we have the site as i
                    #find index
                    #site_off_i = self.struct.sites.index(nearest)
                        
                    #we have already done this so move along, checking diag
#                    if H[interactions*site_off_i, interactions*site_off_i]\
#                    != 0.0:
#                        continue
                    
                    #now add components
                    #ex s1, with s2, px2, py2, pz2
                    H_off = np.matrix([
    [V_ss * cur_g[0], V_sp*cur_g[1], V_sp*cur_g[2], V_sp*cur_g[3]],
    [-1*V_sp*cur_g[1], V_xx*cur_g[0], V_xy*cur_g[3], V_xy*cur_g[2]],
    [-1*V_sp*cur_g[2], V_xy*cur_g[3], V_xx*cur_g[0], V_xy * cur_g[1]],
    [-1*V_sp*cur_g[3], V_xy*cur_g[2], V_xy*cur_g[1], V_xx*cur_g[0]] ])
                                 
                    if lat_fac == 1:
                        #top right
                        H[interactions*site_i:interactions*(site_i+1),
                interactions*site_off_i:interactions*(site_off_i+1)] = H_off
                        #now do the opposite side of the hamiltonian, bottom left
                        H[interactions*site_off_i:interactions*(site_off_i+1),
                          interactions*site_i:interactions*(site_i+1)] =\
                              H_off.conjugate().T
                    else: #now we have to do the reverse as above H
                        #top right
                        H[interactions*site_i:interactions*(site_i+1),
                interactions*site_off_i:interactions*(site_off_i+1)] =\
                        H_off.conjugate().T
                        #now do the opposite side of the hamiltonian, bottom
                        H[interactions*site_off_i:interactions*(site_off_i+1),
                          interactions*site_i:interactions*(site_i+1)] =\
                              H_off
                    #example: for 16 atom basis, first (4x8) columns are
                    #sublattice 1, so they are the not conjugated
                              
#                    except ValueError: #i.e. it isn't in the list
#                        #check if periodic image.
#                        if [nearest.is_periodic_image(i) for i in self.struct.sites]:
#                            #so if it is true, we have the site as i
#                            #find index
#                            site_off_i = self.struct.sites.index(i)
#                            
#                        #don't worry about the hamiltonian, move along
#                        #continue
            
                site_i += 1
#Generate Hamiltonian for a specific kpt.
#------------------------------------------------------ 
            
            #find eigenvalues for each kpt
            eig_list.append(np.linalg.eigvalsh(H))
            
#        print eig_list
        eig_list = np.array(eig_list)
#        print H == H.H
        self.eig = eig_list
        #print H == H.H
        #print H == H.H
        return eig_list, k_points
             
        
if __name__ == '__main__':
    
    E_s = 0
    E_p = 7.2
    V_ss = -8.13/4
    V_sp = 5.88/4
    V_xx = 3.17/4
    V_xy = 7.51/4

    a = LCAO('/Users/cpashartis/Box Sync/Masters/LCAO_tightbinding/test_cifs/POSCAR.Si_16atom')
    #a.load_struct()  #'Si_1atom.cif'
    eig, kpt = a.build_diamond_nn_H(a.diamond_g, a.k_points(), (E_p,V_ss,V_sp,V_xx, V_xy) )
    print eig
    f = plt.figure()
    ax = f.add_subplot(111)
    x = np.linspace(0,1,eig.shape[0])
    for i in range(eig.shape[1]):
        ax.plot(x, eig[:,i])
#labels = ['L', r'$\Gamma$', 'X', r'$\Gamma$']
#plt.xticks(bin_edges, labels)
#a = plt.gca()
#lims = a.get_ylim()
#plt.vlines(bin_edges[1:-1],lims[0],lims[1], color = 'k', linestyle = '--')
    ax.set_title('Silicon Tight Binding')
    ax.set_ylabel('Energy (eV)')
    #ax.set_ylim(-10, 15)
    f.savefig('Silicon_TB.pdf', dpi = 1000 )
    plt.close('all')