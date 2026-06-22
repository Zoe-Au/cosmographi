import os
from warnings import filters
from caskade import Param, forward
from cosmographi.cosmology import Cosmology
from .base import TransientSource
from typing import Any
import pickle
import numpy as np
from agnSED.photometry import Photometric
from ..utils.constants import c_nm, c_m, G, sigma, h_m2kg, k_m2kgsminus2
from .func import temperature, integrand_funct
from jax.scipy.integrate import trapezoid
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


ABSOLUTE_FLUX_TO_LUM_DENSITY = (10 * 3.086e+16) **2 * jnp.pi * 4


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


class AGNSourceThinDisk(TransientSource):
    """ create an AGN source whose SED is modelled according to the thin disk model

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
    inclination_angle: Param(float)
        Inclination angle of the AGN with respect to the observer. The oberver's line of sight makes an angle i 
        to the normal to the disc plane. In radians. 
    r_star: Param(float)
        inner radius of the thin disk, smallest radius from which heat radiates. We assume it to be the innermost 
        stable circular orbit (ISCO), since we assume a non-rotating black hole.
    times: jnp.ndarray[float] | None
        A vector of length equal to <time_perturbations> whose values equal to the times represented by <time_perturbations>.
    time_perturbations: jnp.ndarray[float] | None
        An array containing all of the values the base luminosities will be multiplied by to cause variability with respect to time. 
        It slength will determine the number of timepoints we will have in the grid. 
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float 
    accretion_rate: float
    inclination_angle: float
    r_star: float
    times: jnp.ndarray
    perturbations: jnp.ndarray

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, accretion_rate: float = None, inclination_angle: float = None, 
                 r_star: float = None, perturbations_ar: jnp.ndarray = None, start_time: float = None, end_time: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="solar masses")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="dimensionless")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.r_star = Param("r_sar", r_star, units="metres", description="inner radius of the thin disk, smallest radius from which heat radiates")
        self.perturbations = Param("perturbations", perturbations_ar, units="dimentionless", shape=(len(perturbations_ar), ), description="a vector containing all of the numbers the base luminosities will be " \
        "multiplied by. Cause the variability with respect to time. Its length will determine the number of time points queried.")
        self.times = jnp.linspace(start_time, end_time, len(perturbations_ar))
    
    @forward
    def base_luminosity_density(self, w: float, num_integration_points: int = 1000, inclination_angle=None, blackhole_mass=None, accretion_rate=None, r_star=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in Nanowatts * s"""
        freq = c_m/w
        # we assume R∗​=RISCO​=6GM​/c**2
        first_prod = 4*jnp.pi*h_m2kg*jnp.cos(inclination_angle)*freq**3/c_m**2 / (10 * 3.086e+16)**2
        # assume R_out is 10**4 * r_star
        r_seq = jnp.linspace(r_star, 10**4 * r_star, num_integration_points)
        y_seq = integrand_funct(R=r_seq, freq=freq, mass=blackhole_mass, acc_rate=accretion_rate, r_star=r_star)
        second_prod = trapezoid(y_seq, r_seq)
        flux = first_prod * second_prod 
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density
    
    @forward
    def luminosity_density(self, w: jnp.ndarray, t: jnp.ndarray, integration_points: int = 1000, perturbations=None) -> jnp.ndarray:
        """ Interpolate the luminosity density for a all combinations of the elements in <w> and <t>."""
        base_luminosity = jax.vmap(self.base_luminosity_density, in_axes=(0, None))(w, integration_points)
        perturbation = jnp.interp(t, self.times, perturbations) # linearly interpolate perturbations
        return base_luminosity[None, :] * perturbation[:, None]