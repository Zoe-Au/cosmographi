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
    edd_ratio: float. 
        Eddington ratio of the black hole #TODO: add units.
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: Param
    edd_ratio: Param

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, 
                 edd_ratio: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="solar masses")
        self.edd_ratio = Param("edd_ratio", edd_ratio, shape=(), 
                                   description="Eddington ratio of the black hole", units="dimensionless")    
    
    @forward
    def luminosity_density(self, z, t: float, w: jnp.ndarray) -> jnp.ndarray:
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
        t: the time at observation (MJD)

        Returns
        -------
        jnp.ndarray
            Luminosity density array.
        """
        if not z == 0.1:
            raise NotImplementedError("The current AGN model is only for redshift 0.1. Please choose z=0.1.")
        if not jnp.all(338 <= w <= 395):
            raise NotImplementedError("The curent AGN model is only for passband u. Please choose a value between " \
            "338 and 395 nm.")
        flux = self.get_flux(w, t)
        flux_W_per_m2_Hz = flux * 10 ** (-35)
        distance_parsec = self.cosmology.luminosity_distance(z) * Mpc_to_cm 
        distance_metre = distance_parsec * 3.086e+16
        luminosity_density = flux_W_per_m2_Hz * 4 * jnp.pi * distance_metre**2
        return luminosity_density

    def get_flux(self, w: jnp.ndarray, t: float) -> jnp.ndarray:
        """
        Compute the flux of the AGN source at a given wavelength and time.

        Parameters
        ----------
        w: jnp.ndarray
            Wavelength array (nm) of the rest frame.
        t: float
            Time of observation (MJD).

        Returns
        -------
        jnp.ndarray
            Flux array in units of nJy
        """
        if not jnp.all((338 <= w) & (w <= 395)):
            raise NotImplementedError("The curent AGN model is only for passband u. Please choose a value between " \
            "338 and 395 nm.")
        a, c, d = self.get_parameters()
        return a * jnp.sin(t - c) + d

    def get_parameters(self) -> list[float]:
        """ Get the parameters of the sine function based on the black hole mass and eddington ratio. """
        bh_mass = self.blackhole_mass.value
        edd_ratio = self.edd_ratio.value
        a =  -452437 * bh_mass - 701078 * edd_ratio + 3432715
        c = 88 * bh_mass + 32186 * edd_ratio - 6456
        d = 303617 * bh_mass + 2608280 * edd_ratio - 2351906
        return [a, c, d]






