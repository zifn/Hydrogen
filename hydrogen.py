# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:21:16 2018

@author: Main
model of third order susptability in hydrogen
"""
from time import strftime
import itertools
import pickle
import numpy as np
import sympy as sp
from sympy import init_printing
from sympy.functions import conjugate
from sympy.physics.hydrogen import R_nl
from sympy.physics.hydrogen import E_nl
from sympy.functions.special.spherical_harmonics import Ynm
import scipy.constants as constants
init_printing()

def hydrogen_omega_matrix(basis):
    """
    Uses sympy to calculate and return the matrix representation of the omega matrix, the matrix of
    einstien frequencies in atomic units hbar = m_e = e = 1
    
    Parameters
    ----------
    basis: iterable of tuples
        each tuple defines a state of hydrogen to use in calculating the matrix. The number of tuples
        is the number of rows and columns in the returned matrices. Each tuple has 3 entries (n, l, m)
        where n is the principle quantum number, l is the total angular momentum quantum number, and
        m is the magnetic quantum number.
        
    Returns
    -------
    sp.Matrix
        This is the omega matrix for hydrogen given the basis set used
    """
    E_basis = [E_nl(n) for n, l, m in basis]
    return sp.Matrix([[E_f - E_i for E_f in E_basis] for E_i in E_basis])

def hydrogen_dipole_matrix(basis):
    """
    Uses sympy to calculate an return the matrix representation of the dipole vector operator in the
    hydrogen basis set in atomic units for x, y, z coordinates.
    
    Parameters
    ----------
    basis: iterable of tuples
        each tuple defines a state of hydrogen to use in calculating the matrix. The number of tuples
        is the number of rows and columns in the returned matrices. Each tuple has 3 entries (n, l, m)
        where n is the principle quantum number, l is the total angular momentum quantum number, and
        m is the magnetic quantum number.
        
    Returns
    -------
    list of matrices
        Returns 3 sp.Matrix objects representing the dipole vector operator in the x, y, and z
        cartesian coordinates respectively 
    """
    # define vars
    r, theta, phi = sp.symbols("r, theta, phi", real=True)
    
    # define transformations to cartiesian and the dipole operators
    z = r*sp.cos(theta)
    x = r*sp.sin(theta)*sp.cos(phi)
    y = r*sp.sin(theta)*sp.sin(phi)
    
    # define basis in atomic units
    print("making basis wavefunctions")
    H_nlm = [(R_nl(n, l, r)*sp.simplify(Ynm(l, m, theta, phi).expand(func=True)), (n, l, m))
             for n, l, m in basis]
    
    # find dipole transition matrix elements: H_init = inital hydrogen orbital, H_final = final hydrogen orbital 
    mu = []
    element = lambda H_final, op, H_init: sp.integrate(H_final*op*H_init*sp.sin(theta)*r**2, (r, 0, sp.oo), (theta, 0, sp.pi), (phi, 0, 2*sp.pi))
    select_rules = lambda basis_f, basis_i: True#all([abs(basis_f[1] - basis_i[1]) == 1, abs(basis_f[2] - basis_i[2]) >= 1])
    print("Computing matrix elements")
    for mu_oper in [x, y, z]:
        mu.append(sp.Matrix([[element(conjugate(bra), mu_oper, ket) 
                              if select_rules(basis_f, basis_i) else 0
                              for bra, basis_f in H_nlm] 
                              for ket, basis_i in H_nlm]))
    return mu

def decay_rate_estimates(omega, dipole_vector, alpha=1/137.035999139, c=137.035999139, n=1):
    """
    Estimates the decay rate matrix (gamma matrix) using expressions for spontaneous emission rates
    given the omega matrix, and dipole_vector matrix. Assumes hartree atomic units.
    
    Parameters
    ----------
    omega: sympy Matrix
        the matrix of einstien frequencies for a given transition between two states
    dipole_vector: tuple of three sympy Matrices
        the matrix representation of the dipole vector operator being used
    alpha: float
        The fine structure constant
    c: float
        The speed of light
    n: float
        The index of refraction
        
    Returns
    -------
    sympy Matrix
        The decay rate matrix (gamma matrix) using the rate of spontaneous emission as the estimator
    """
    dipole = lambda m, n: sum([abs(dipole_vector[xyz][m, n])**2 for xyz in range(3)])
    
    return sp.Matrix([[(4*alpha*n*dipole(j, i)*omega[j, i]**3)/(3*c**2)
                       for i in range(omega.shape[1])] # col index
                       for j in range(omega.shape[0])]) # row index

def chi_3(k, i, j, h, omw_r, omw_q, omw_p, dipole, relax_gamma, density_matrix, omega_matrix):
    """
    Computes elements of the 2nd hyperpolarizability using Boyd's quantum discription of chi3
    (Nonlinear Optics 3rd ed pg 182).
    """
    numb_of_states = density_matrix.shape[0]
    cartiesian_indices = [i, j, h]
    lights_omegas = [omw_r, omw_q, omw_p]
    chi3_kijk = 0
    
    def chi_3_no_permuations(xyz, omegas, numb_states):
        chi3_temp = 0
        for n in range(numb_states):
            for m in range(numb_states):
                for v in range(numb_states):
                    for l in range(numb_states):
                        a = (density_matrix[m, m] - density_matrix[l, l]) \
                            *(dipole[k][m, n]*dipole[xyz[1]][n, v]*dipole[xyz[0]][v, l]*dipole[xyz[2]][l, m]) \
                            /((omega_matrix[n, m] - omegas[2] - omegas[1] - omegas[0] - sp.I*relax_gamma[n, m]) \
                             *(omega_matrix[v, m] - omegas[2] - omegas[1] - sp.I*relax_gamma[v, m]) \
                             *(omega_matrix[l, m] - omegas[2] - sp.I*relax_gamma[l, m]))
    
                        b = (density_matrix[l, l] - density_matrix[v, v]) \
                            *(dipole[k][m, n]*dipole[xyz[1]][n, v]*dipole[xyz[0]][l, m]*dipole[xyz[2]][v, l]) \
                            /((omega_matrix[n, m] - omegas[2] - omegas[1] - omegas[0] - sp.I*relax_gamma[n, m]) \
                             *(omega_matrix[v, m] - omegas[2] - omegas[1] - sp.I*relax_gamma[v, m]) \
                             *(omega_matrix[v, l] - omegas[2] - sp.I*relax_gamma[v, l]))
    
                        c = (density_matrix[v, v] - density_matrix[l, l]) \
                            *(dipole[k][m, n]*dipole[xyz[1]][v, m]*dipole[xyz[0]][n, l]*dipole[xyz[2]][l, v]) \
                            /((omega_matrix[n, m] - omegas[2] - omegas[1] - omegas[0] - sp.I*relax_gamma[n, m]) \
                             *(omega_matrix[n, v] - omegas[2] - omegas[1] - sp.I*relax_gamma[n, v]) \
                             *(omega_matrix[l, v] - omegas[2] - sp.I*relax_gamma[l, v]))
    
                        d = (density_matrix[l, l] - density_matrix[n, n]) \
                            *(dipole[k][m, n]*dipole[xyz[1]][v, m]*dipole[xyz[0]][l, v]*dipole[xyz[2]][n, l]) \
                            /((omega_matrix[n, m] - omegas[2] - omegas[1] - omegas[0] - sp.I*relax_gamma[n, m]) \
                             *(omega_matrix[n, v] - omegas[2] - omegas[1] - sp.I*relax_gamma[n, v]) \
                             *(omega_matrix[n, l] - omegas[2] - sp.I*relax_gamma[n, l]))
                        chi3_temp += a - b - c + d
        return sp.simplify(chi3_temp)
    
    simultaneous_permutations = list(itertools.permutations(zip(cartiesian_indices, lights_omegas)))
    numb_aves = len(simultaneous_permutations)
    for current_xyzs_omegas in simultaneous_permutations:
        current_xyzs = [xyz_omega[0] for xyz_omega in current_xyzs_omegas]
        current_omegas = [xyz_omega[1] for xyz_omega in current_xyzs_omegas]
        chi3_kijk += chi_3_no_permuations(current_xyzs, current_omegas, numb_of_states)/numb_aves
    return sp.simplify(chi3_kijk)

def hydrogen_basis(max_n):
    """
    n
    l = [0,n-1]
    m = [-l, l]
    """
    basis = []
    for n in range(1, max_n + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                basis.append((n, l, m))
    return basis

def pickle_matrices(object_to_pickle, file_path):
    """
    Uses the Pickle module to save computationally expensive matrices to a given
    file path
    """
    with open(file_path, 'wb') as f:
        pickle.dump(object_to_pickle, f)
        
def unpickle_matrices(file_path):
    """
    Uses the Pickle module to save computationally expensive matrices to a given
    file path
    """
    with open(file_path, 'rb') as f:
        dipole, omega, decay = pickle.load(f)
    return dipole, omega, decay

def perturb_density(rho_init, mu, omega, decay, efield_pump, time_var, t_finals, hbar = 1):
    """
    Uses the density matrix formulation of first order perturbation theory with
    damping to calculate the density matrix after some time (t) in atomic units
    (boyd 2nd ed eq 3.5.1 pg 161 [pg 173 pdf])
    
    Parameters
    ----------
    rho_init: n by n sympy matrix 
    mu: list of three n by n sympy matrices
    omega: n by n sympy matrix 
    decay: n by n sympy matrix 
    efield_pump: list of three sympy functions
    time: float
    hbar: 1 in atomic units
    
    Notes
    -----
    integration starts at t = 0
    """
    size_row, size_col = rho_init.shape
    rho_final = sp.zeros(size_row, size_col)
    V = sp.zeros(size_row, size_col)
    t_prime = sp.symbols("t_prime")
    t = time_var
    #print("E_field = {0}\nDipole = {1}".format(efield_pump, mu))
    for element in [efield_pump[axis]*mu[axis] for axis in range(len(mu))]:
        V += element
    print("Dipole Potential = {}".format(V))
    commute = ((V*rho_init - rho_init*V)*(-sp.I/hbar)).subs({t: t_prime})
    print("[V, rho_init] = {}".format(commute))
    
    final_rhos = []
    for t_final in t_finals:
        for r in range(size_row): 
            for c in range(size_col):
                exp_factor_integral = sp.exp((sp.I*omega[r, c] + decay[r, c])*t_prime)
                exp_factor = sp.exp((-sp.I*omega[r, c] + decay[r, c])*t)
                integrand = commute[r, c]*exp_factor_integral
                try:
                    integral = sp.integrate(integrand, (t_prime, 0, t_final),  risch=False)
                except Exception as e:
                    print("integrand = {}".format(integrand))
                    print("row = {0}, col = {1}".format(r, c))
                    raise(e)
                rho_final[r, c] = integral*exp_factor.subs({t: t_final}) + rho_init[r, c]
        final_rhos.append(rho_final)
        print("computed perturbed rho for t_final = {}".format(t_final))
        print("perturbed rho\n-------------\n{}".format(rho_final))
        
    return final_rhos
    

def main():
    """
    for calculating estimates for the relaxation matrix consider using these eqs
    https://en.wikipedia.org/wiki/Spontaneous_emission#Rate_of_spontaneous_emission
    """
    chi3_conversion_factor = constants.physical_constants["atomic unit of 2nd hyperpolarizability"][0]
    basis = hydrogen_basis(2)
    pickled_output_file = "pickled_dipole_freq_decay_with_{}_basis_functions_{}.pickle".format(len(basis), strftime("%H-%M-%S"))
    
    omw = (E_nl(2) - E_nl(1))/8 # driving and probing light freq
    density_0 = sp.zeros(len(basis), len(basis))
    density_0[0, 0] = 1
    
    #make pump field
    #see https://en.wikipedia.org/wiki/Gaussian_function
    # I = 0.5*c*n*epsilon_0*|E|**2
    # |E| = sqrt(2*I/c*n*epsilon_0)
    time = sp.symbols("time")
    time_conversion = constants.physical_constants['atomic unit of time'][0]/constants.femto
    efield_conversion = constants.physical_constants["atomic unit of electric field"][0]
    t_0 = 200/time_conversion #fs
    FWHM_durration = 100/time_conversion #fs
    intensity = 2*10**20 #W/cm**2
    n = 1
    E_mag = np.sqrt(2*(intensity*10**4)/(constants.c*n*constants.epsilon_0))/efield_conversion
    E_x = 0
    E_y = 0
    E_z = E_mag*sp.exp((-4*np.log(2)*(time - t_0)**2)/(FWHM_durration**2))
    E_field = [E_x, E_y, E_z]
    
    print("hydrogen basis set = {}".format(basis))
    print("Initial density matrix")
    sp.pprint(density_0)
    print("computing hydrogen\'s dipole matrix")
    mu = hydrogen_dipole_matrix(basis)
    print("computing hydrogen\'s frequency matrix")
    omega = hydrogen_omega_matrix(basis)
    print("estimating hydrogen\'s decay rate matrix")
    gamma = decay_rate_estimates(omega, mu)
    
    save_data_dict = {"mu": mu, "omega": omega,"decay": gamma}
    
    time_steps = np.linspace(0, 1000, 6) #fs
    print("time steps = {} fs".format(time_steps))
    
    save_data_dict["time_steps_fs"] = time_steps
    
    perturbed_rhos = perturb_density(density_0, mu, omega, gamma, E_field, time, time_steps/time_conversion)
    save_data_dict["perturbed_rhos"] = perturbed_rhos
    print("perturbed_rhos = {}".format(perturbed_rhos))

    print("computing chi3_yxxy and chi3_yxyx")
    chi3_yxxy = np.array([chi_3(1, 0, 0, 1, omw, omw, omw, mu, gamma, rho, omega) for rho in perturbed_rhos])
    save_data_dict["chi3_yxxy_au"] = chi3_yxxy
    print("chi3_yxxy = {} a.u.".format(chi3_yxxy))
    chi3_yxyx = np.array([chi_3(1, 0, 1, 0, omw, omw, omw, mu, gamma, density_0, omega) for rho in perturbed_rhos])
    save_data_dict["chi3_yxyx_au"] = chi3_yxyx
    print("chi3_yxyx = {} a.u.".format(chi3_yxyx))
    chi3_eff = chi3_yxxy + chi3_yxyx
    save_data_dict["chi3_eff_SI"] = chi3_eff*chi3_conversion_factor
    print("chi3_eff = {} m^2/V^2".format(chi3_eff*chi3_conversion_factor))
    print("|chi3_eff|^2 = {} SI".format(abs(chi3_eff*chi3_conversion_factor)**2))
    
    pickle_matrices(save_data_dict, pickled_output_file)
    

if __name__ == "__main__":
    main()
