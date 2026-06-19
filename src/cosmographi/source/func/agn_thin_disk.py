from ...utils.constants import G, sigma, h_m2kg, k_m2kgsminus2
import jax.numpy as jnp

def temperature(R: float, mass: float, acc_rate: float, r_star: float) -> float:
    """ Return the temperature at distance <R> for an AGN of <mass> and accretion rate <acc_rate>"""
    first_prod = 3 * G * mass * acc_rate / 8 /jnp.pi / R**3 / sigma
    second_prod = 1 - (r_star / R) ** (1/2)
    return (first_prod * second_prod) ** (1/4)

def integrand_funct(R: float, freq: float, mass: float, acc_rate: float, r_star: float) -> float:
    return R / (jnp.exp(h_m2kg*freq/k_m2kgsminus2/temperature(R, mass, acc_rate, r_star)) - 1)

