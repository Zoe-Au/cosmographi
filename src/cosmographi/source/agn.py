import os
from warnings import filters

import jax.numpy as jnp
import jax
from caskade import Param, forward
import numpy as np
from cosmographi.cosmology import Cosmology
from .base import TransientSource
from ..utils import flux
from ..utils.constants import Mpc_to_cm
from typing import Any
import tinygp

from lightcurvelynx.models.agn import AGN

def delta_fun(i, j) -> int:
    if i == j:
        return 1
    return 0


class AGNSource(TransientSource):
    """
    An AGN source model generated through damped random walk.

    Parameters
    ---------- 
    cosmology: Cosmology. 
        Optional. If given, it can be used to compute the distance modulus from the redshift.
    name: str. 
        Optional. Name of the source.
    blackhole_mass: float. 
        Mass of the black hole in solar masses #TODO: change units
    accretion_rate: float. 
        Accretion rate of the black hole #TODO: add units. Maybe switch to Eddington ratio?
    """
    cosmology: Cosmology
    name: str
    # blackhole_mass: float
    # accretion_rate: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, 
                 edd_ratio: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="solar masses")
        self.edd_ratio = Param("edd_ratio", edd_ratio, shape=(), 
                                   description="Eddington ratio of the black hole", units="dimensionless")    
    

    def likelihood_function(self) -> float:
        """
        Compute the likelihood of the model.

        Parameters
        ----------

        Returns
        -------
        float
            The computed likelihood value.
        """

        
        

    def _create_covariance_matrix(self, tau: float, sigma: float, t0: float, t: jnp.ndarray) -> jnp.ndarray:
        """
        Create the covariance matrix for the damped random walk model.

        Parameters
        ----------
        tau: float
            Characteristic timescale of the variability (in days).
        sigma: float
            Variability amplitude (in magnitudes).
        t0: float
            Reference time (in days).
        t: jnp.ndarray
            Time array (in days).

        Returns
        -------
        jnp.ndarray
            Covariance matrix.
        """
        abs_dt = jnp.abs(t[:, None] - t0)
        k = sigma**2 * jnp.exp(-abs_dt / tau)
        num_obs = len(t)
        return jnp.from_function(k + sigma**2 * self._delta_fun, (num_obs, num_obs), dtype=jnp.float32)



    @forward
    def luminosity_density(self, w, p, t0, x1, c, z) -> jnp.ndarray:
        """
        Compute the luminosity density of the AGN source at a given wavelength and redshift in units of
        erg/s/nm and time in units of seconds.

        Note: This implementation is a WIP. 

        Parameters
        ----------
        w: jnp.ndarray
            Wavelength array (nm) of the rest frame.
        z: float
            Redshift of the source.

        Returns
        -------
        jnp.ndarray
            Luminosity density array.
        """








