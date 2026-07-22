import jax.numpy as jnp
c_m = 299792458  # speed of light in m/s
c_km = c_m / 1000  # speed of light in km/s
c_nm = c_m * 1e9  # speed of light in nm/s
c_cm = c_m * 1e2  # speed of light in cm/s
Mpc_to_cm = 3.085677581491367e24  # 1 Mpc in cm
k_si = 1.380649e-23 # Planck constant in J*s
h = jnp.asarray(6.62607015e-27, dtype=jnp.float64)  # Planck constant in erg*s
k = jnp.asarray(1.380649e-16, dtype=jnp.float64)  # Boltzmann constant in erg/K
G = jnp.asarray(6.6743e-11, dtype=jnp.float64)  # Gravitational constant in m3⋅kg−1⋅s−2
sigma =  jnp.asarray(5.670374419e-8, dtype=jnp.float64) #  Stephan-Boltzman constant (sigma)in W⋅m-2⋅K-4
h_m2kg = jnp.asarray(6.62607015e-34, dtype=jnp.float64) # Planck's constant in m2 kg / s 
k_m2kgsminus2 = jnp.asarray(1.380649 * 10e-23, dtype=jnp.float64) # Boltzmann constant in m2 kg s-2 K-1  
M_sun = jnp.asarray(1.98847e30, dtype=jnp.float64)      #Mass of the sun in  kg
me_g = 9.1093837e-28 # electron mass in grams
