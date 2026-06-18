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
import pickle
import agnSED
from agnSED.photometry import Photometric
from ..utils.constants import c_nm



class AGNSource(TransientSource):
    """ An AGN source. WIP Model, do not use.
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
        if not z == 0.000000002369:
            raise NotImplementedError("The current AGN model is only for redshift 0.000000002369. Please choose z=0.000000002369.")
        if not jnp.all(338 <= w <= 395):
            raise NotImplementedError("The curent AGN model is only for passband u. Please choose a value between " \
            "338 and 395 nm.")
        flux = self.get_flux(w, t)
        flux_W_per_m2_Hz = flux * 10 ** (-35)
        distance_parsec = 10
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
        t: jnp.ndarray
            Time of observation (MJD).

        Returns
        -------
        jnp.ndarray
            Flux array in units of nJy
        """
        if not jnp.all((338 <= w) & (w <= 395)):
            raise NotImplementedError("The curent AGN model is only for passband u. Please choose a value between " \
            "338 and 395 nm.")
        a,b, c, d = self.get_parameters()
        return jnp.exp(a * jnp.sin( b * (t - c)) + d)

    def get_parameters(self) -> list[float]:
        """ Get the parameters of the sine function based on the black hole mass and eddington ratio. """
        bh_mass = self.blackhole_mass.value
        edd_ratio = self.edd_ratio.value
        a =  -3.6379 * bh_mass + 0.953 * edd_ratio + 0.8349
        b =  0.0 * bh_mass + 0.0001 * edd_ratio + 0.0002
        c = 0.4285 * bh_mass + 3.3066 * edd_ratio -2.3468
        d = 2.5053 * bh_mass - 1.7961 * edd_ratio + 0.2613
        return [a, b, c, d]


class AGNSourceAGNFitter(TransientSource):
    """ An AGN source. WIP Model, do not use.
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
    sfh_tau:
        Star formation tau parameter
    nh_value:
        NH collumn density of the torus.
    irlum:
        cold dust emission parametrization
    """
    name: str
    cosmology: Cosmology
    logBHmass: float
    logEddra: float
    tau: float
    Nh: float
    irlum: float
    age: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, 
                 edd_ratio: float = None, sfh_tau: float = None, nh_value: float = None, 
                 irlum: float = None, age: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.logBHmass = blackhole_mass
        self.logEddra = edd_ratio
        self.tau = sfh_tau
        self.Nh = nh_value
        self.irlum = irlum
        self.age = age
    
    def luminosity_density(self, z: float, t: float, w: jnp.ndarray) -> jnp.ndarray:
        """ Compute the luminosity density of the AGN source at a given wavelengths <w> and time <t>
        """
        w_res = w/(z + 1)
        with open("knn_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        params = self._get_param_values_nan()
        return pipeline.predict(w_res + params)

    def _get_param_values_nan(self) -> list:
        """Return the parameter values of this AGN, wogh missing ones replaced by jnp.nan"""
        params = [self.logBHmass, self.logEddra, self.age, self.tau, self.irlum, self.Nh]
        for i in range(len(params)):
            if params[i] in None:
                params[i] = jnp.nan
        return jnp.ndarray(params)


class AGNSourceTong2026(TransientSource):
    """ create an AGN source whose SED is modelled according to 
    https://iopscience.iop.org/article/10.3847/1538-4357/ae41bd/pdf 

    Parameters
    ---------- 
    cosmology: Cosmology. 
        Optional. If given, it can be used to compute the distance modulus from the redshift.
    name: str. 
        Optional. Name of the source.
    blackhole_mass: float. 
        Mass of the black hole in solar masses 
    accretion_rate: float. 
        Accretion rate of the black hole 
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float
    accretion_rate: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, 
                 accretion_rate: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        # self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
        #                            description="Mass of the black hole", units="solar masses")
        # self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
        #                            description="accretion_rate of the black hole", units="dimensionless")  
        self.blackhole_mass = blackhole_mass
        self.accretion_rate = accretion_rate
        
    # @forward
    def luminosity_density(self, start: float, end: float, z: float) -> None:
        """ Return the luminosity density between wavelengths <start> and <end> and at redshift <z>.
        #TODO include time (+ perturbations?)
        Note: Remeber agnSED takes it in the frecuency domain

        Preconditions:
             - <start> and <end> are assumed to be in nm
        """
        # transforming the wavelengths to at-observer wavelengths (accounting for redshift)
        start_source = start/(z + 1)
        end_source = end/(z + 1)
        start_source_freq = c_nm / start_source
        end_source_freq = c_nm / end_source
        params = np.asarray([[np.log10(self.blackhole_mass), np.log10(self.accretion_rate)]])
        lognu0, lognu1 = sorted([np.log10(start_source_freq), np.log10(end_source_freq)])
        return Photometric(params, lognu0, lognu1)

        

    

