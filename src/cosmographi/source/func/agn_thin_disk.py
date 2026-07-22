from ...utils.constants import G, sigma, h, k_m2kgsminus2, c_m, M_sun, k, h_m2kg
from ..func.blackbody import blackbody_luminosity_density
import jax.numpy as jnp

SECS_IN_YEAR = 365.25 * 24 * 3600 

def thin_disk_temperature(R: float, mass: float, acc_rate: float, r_star: float) -> float:
    """ Return the temperature at distance <R> for an AGN of <mass> and accretion rate <acc_rate>"""
    first_prod = 3 * G * mass * acc_rate / 8 /jnp.pi / R**3 / sigma
    second_prod = 1 - (r_star / R) ** (1/2)
    return (first_prod * second_prod) ** (1/4)

def integrand_funct(R: float, freq: float, mass: float, acc_rate: float, r_star: float) -> float:
    return R / (jnp.exp(h_m2kg*freq/k_m2kgsminus2/thin_disk_temperature(R, mass, acc_rate, r_star)) - 1)



################# SU 2026 ##################

f = 1
alpha = 0.1
beta = 0.95

def slim_disk_corona_temperature(R: float, mass: float) -> float:
    """Return the temperature (in K) at radius <R> of an AGN of mass <mass>  (in solar masses)."""
    # slim_disk
    scwarzchild_r = mass * 2 * G / c_m**2
    r = R / scwarzchild_r
    beta = 1 - jnp.sqrt(3/r)
    beta = jnp.clip(beta, 1e-12, None)
    bh_msun = mass/M_sun
    radiation_temp = 4.965 * 10**7 * beta**(1/8) * f**(1/8) * bh_msun**(-1/4) * r**(-1/2)
    # corona
    # assume l = 10R_s
    # gas_temp = 4.86 * 10**9 * alpha**(-9/80) * beta**(-1/2) * (mass/(10**8 * M_sun))**(1/80) * (edd_normalized_acc_rate/0.1)**(1/10) * (r/10)**(-51/160)  
    return radiation_temp 

def slim_disk_corona_integrand_funct(R:float, mass: float, freq: float) -> float:
    return R / (jnp.exp(h_m2kg*freq/k_m2kgsminus2/slim_disk_corona_temperature(R, mass=mass)) - 1)

def F(theta: float) -> float:
    """This assumes theta_e < 1, which is the cse for the assumed fixed Te"""
    return 4 * (2*theta/jnp.pi**3)**(1/2) * (1 + 1.781 * theta**1.24) + 1.73 * theta**(3/2) * (1 + 1.1 * theta + theta**2 - 1.25*theta**(5/2))


################ Lipunova 2018 ######################
def compute_T0(M: float, M_dot: float, r_in: float) -> float:
    """ Compute the effective temperature at T0, which is equal to the maximum effective temperature at the disk surface.
    Computed as done in lightcurvelynx and https://doi.org/10.1007/978-3-319-93009-1_1.

    Parameters
    ----------
    M : float
        Mass of the black hole in kg.
    M_dot : float
        Accretion rate in kg/year.
    r_in : float
        Inner radius of the disk in m.

    Returns
    -------
    float
        Effective temperature at T0 in Kelvin.
    """
    M_dot_s = M_dot / SECS_IN_YEAR 
    return 2**(3/4) * (3/7)**(7/4) * (G * M * M_dot_s / (jnp.pi * sigma * r_in**3))**(1/4)


def standard_disk_integration_f(x: float) -> float:
    """Function that must be integrated for Lipunova 2018."""
    return  x**(5/3) / (jnp.expm1(x))
