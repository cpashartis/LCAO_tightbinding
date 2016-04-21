# LCAO_tightbinding

##Background
LCAO tightbinding makes use of the fomalism introduced by *Chadi and Cohen (1975)*. LCAO or Linear Combination of Atomic Orbitals
allows us to build a tightbinding model knowing the orbital overlap parameters between two atoms. For each type of orbital interaction
there is a corresponding energy associated with it that can be found using *bond decomposition techniques* (see Tightbinding Model of 
Electronic Structures, http://cacs.usc.edu/education/phys516/04TB.pdf)

##Usage
The code makes use of the python package pymatgen to be able to generically solve tightbinding. If you provide any typical structure
file it *should work* (let me know if it doesn't). I'm currently in the process of upgrading it with a database of tight binding parameters
for use.

Currently only supports s, and p orbital nearest interactions
###For now:###
    E_s = 0
    E_p = 7.2
    V_ssig = -2.032
    V_sp = 2.546
    V_ppSig = 4.5475
    V_ppPi = -1.085

    a = LCAO('/Users/cpashartis/Box Sync/Masters/LCAO_tightbinding/test_cifs/POSCAR.Si_16atom')
    eig, kpt = a.build_diamond_nn_H(a.diamond_g, a.k_points(), (E_p,V_ssig,V_sp,V_ppSig, V_ppPi) )
   
0. Import the LCAO module.
1. Simply replace the values of the s-$\Sigma$ bonds, s-p bonds, p-p-$\Sigma$, and p-p-$\Pi$ bonds with the literature values.
2. Also change the file you wish to use, when calling LCAO. The above returns the eigenvalues and kpts.
3. You should be aware that if you wish to have your own symmetry points, you need to provide the function call with an array of
all the kpoints you wish to use (row wise) at *a.k_points()*
