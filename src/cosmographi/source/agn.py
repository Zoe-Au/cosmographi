import os
from warnings import filters
from caskade import Param, forward
from cosmographi.cosmology import Cosmology
from .base import TransientSource
from typing import Any
import pickle
import numpy as np
from agnSED.photometry import Photometric
from ..utils.constants import c_nm, c_cm, c_m, G, h_m2kg, me_g, k, h, k_si
from ..utils.integration import quad
from .func import integrand_funct, slim_disk_corona_integrand_funct, F, compute_T0, standard_disk_integration_f
from jax.scipy.integrate import trapezoid
import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from cosmographi.utils.flux import f_l


PARSECS_TO_METRES = 3.086e+16
ABSOLUTE_FLUX_TO_LUM_DENSITY = (10 * 3.086e+18) **2 * jnp.pi * 4
SECS_IN_YEAR = 31536000
KG_TO_G = 1e3
M_sun = 1.9891e30


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


class AGNSourceSu2026(TransientSource):
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
        A vector of length equal to <time_perturbations> whose values equal to the times represented by <time_perturbations>. In rest frame.
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
                                   description="Mass of the black hole", units="kg")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="dimensionless")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.r_star = Param("r_sar", r_star, units="metres", description="inner radius of the thin disk, smallest radius from which heat radiates")
        self.perturbations = Param("perturbations", perturbations_ar, units="dimentionless", shape=(len(perturbations_ar), ), description="a vector containing all of the numbers the base luminosities will be " \
        "multiplied by. Cause the variability with respect to time. Its length will determine the number of time points queried.")
        self.times = jnp.linspace(start_time, end_time, len(perturbations_ar))
    
    @forward
    def base_luminosity_density(self, w: float, num_integration_points: int = 1000, inclination_angle=None, blackhole_mass=None, accretion_rate=None, r_star=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in erg/s/nm. w must be in nm."""
        freq = c_nm/w
        # we assume R∗​=RISCO​=6GM​/c**2
        first_prod = 4*jnp.pi*h_m2kg*jnp.cos(inclination_angle)*freq**3/c_m**2 / (10 * 3.086e+16)**2
        # assume R_out is 10**4 * r_star
        r_seq = jnp.linspace(r_star, 10**4 * r_star, num_integration_points)
        y_seq = integrand_funct(R=r_seq, freq=freq, mass=blackhole_mass, acc_rate=accretion_rate, r_star=r_star)
        second_prod = trapezoid(y_seq, r_seq)
        flux = f_l(f_nu = first_prod * second_prod *1e3, nu = freq) 
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density
    
    @forward
    def luminosity_density(self, w: jnp.ndarray, t: jnp.ndarray, integration_points: int = 1000, perturbations=None) -> jnp.ndarray:
        """ Interpolate the luminosity density ( erg/s/nm ) for a all combinations of the elements in <w> and <t>."""
        t = jnp.atleast_1d(t)
        base_luminosity = jax.vmap(self.base_luminosity_density, in_axes=(0, None))(w, integration_points)
        perturbation = jnp.interp(t, self.times, perturbations) # linearly interpolate perturbations
        return base_luminosity[None, :] * perturbation[:, None]
    
TRUNCATION_CONSTANT = 17.3 * 0.05**0.07 * 0.95**4.61
n_eff = 0.1
delta = 0.2
beta = 0.95
c3 = 0.3
c1 = 0.5
f = 1 #TODO: look at edge cases
M_sun = 1.9891e30
secs_per_year = 31557600
alpha = 0.1
beta_1 = 1
m_dot_crit = 0.38 * alpha**2.34 * beta**(-0.41)

class AGNSourceSu2026(TransientSource):
    """ create an AGN source whose SED is modelled according to Su et al. 2026 (https://arxiv.org/pdf/2501.10793v2). Mising Corona part!
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
    eddington_normalized_accretion_rate: float
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float 
    accretion_rate: float
    inclination_angle: float
    r_star: float
    times: jnp.ndarray
    perturbations: jnp.ndarray
    eddington_normalized_accretion_rate: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, accretion_rate: float = None, inclination_angle: float = None, 
                 r_star: float = None, perturbations_ar: jnp.ndarray = None, start_time: float = None, end_time: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="kg")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="kg per year")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.perturbations = Param("perturbations", perturbations_ar, units="dimentionless", shape=(len(perturbations_ar), ), description="a vector containing all of the numbers the base luminosities will be " \
        "multiplied by. Cause the variability with respect to time. Its length will determine the number of time points queried.")
        self.times = jnp.linspace(start_time, end_time, len(perturbations_ar))
        self.r_star = Param("r_sar", r_star, units="metres", description="inner radius of the thin disk, smallest radius from which heat radiates")
        self.eddington_normalized_accretion_rate = accretion_rate * KG_TO_G / SECS_IN_YEAR / (1.39e18 * blackhole_mass/M_sun) # dimentionless
        # check conversion
    
    @forward
    def truncated_disk_and_adaf_luminosity_density(self, w: float, num_integration_points: int = 1000, inclination_angle=None, blackhole_mass=None, accretion_rate=None, r_star=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in erg/s/nm. w must be in nm."""
        edd_rate = jnp.clip(self.eddington_normalized_accretion_rate, 1e-12, None)
        freq = c_nm/w
        bh_msun = blackhole_mass/M_sun
        scwarzchild_r = blackhole_mass * 2 * G / c_m**2
        r_star_s = r_star / scwarzchild_r
        r_trunc_s = 17.3 * edd_rate**(-0.886) * TRUNCATION_CONSTANT 
        r_trunc = r_trunc_s * scwarzchild_r
        first_prod = 4*jnp.pi*h_m2kg*jnp.cos(inclination_angle)*freq**3/c_m**2 / (10 * 3.086e+16)**2
        # assume R_out is r_trunc
        r_seq = jnp.linspace(r_star, r_trunc, num_integration_points)
        y_seq = integrand_funct(R=r_seq, freq=freq, mass=blackhole_mass, acc_rate=accretion_rate, r_star=r_star)
        second_prod = trapezoid(y_seq, r_seq)
        flux = f_l(f_nu = first_prod * second_prod *1e3, nu = freq) 
        disk_luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        # ADAF part
        s1 = 1.42 * 10**9 * alpha**(-1/2) * (1-beta)**(1/2) * c1**(-1/2) * c3**(1/2)
        x_m = jnp.exp(3.6 + 1/4 * jnp.log(self.eddington_normalized_accretion_rate))
        s2 = 1.19 * 10**(-13) * x_m
        s3 = 1.05 * 10**(-24)
        # Lum due to cyclosynchroton emission:
        # TODO: We are using this value according to page 8 of the paper. However, it is only valid for high mdot, and this model is for low mdot so it is not ideal. Calculate this value!!
        Te = 2e9
        L_v = s3 * (s1 * s2)**(8/5) * bh_msun**(6/5) * self.eddington_normalized_accretion_rate**(4/5) * Te**(21/5) * freq**(2/5) #ergs s-1 Hz-1
        # Lum due to Bremsstrahlung
        theta_e = k * Te / me_g /c_cm**2
        L_brems = 2.29 * 10**24 * alpha**(-2) * c1**(-2) * jnp.log(r_trunc_s/r_star_s) * F(theta_e) * Te**(-1) * jnp.exp(- h * freq / k/ Te) * bh_msun * self.eddington_normalized_accretion_rate**2
        # lum due to crompton scattering:
        v_p = s1 * s2 * bh_msun**(-1/2) * self.eddington_normalized_accretion_rate**(1/2) * Te**2 * r_trunc_s**(-5/4)
        # assume alpha_c = 0.75
        alpha_c = 0.75
        Lum_cromp = v_p**(-alpha_c) * (s3 * (s1 * s2)**(8/5) * bh_msun**(6/5) * self.eddington_normalized_accretion_rate**(4/5) * Te**(21/5) * v_p**(2/5)) * v_p * freq**(- alpha_c) #ergs s-1 Hz-1
        lum_adaf = L_v + L_brems + Lum_cromp
        lum_per_nm_adaf = lum_adaf * c_nm / w**2
        return lum_per_nm_adaf + disk_luminosity_density
    
    @forward
    def slim_disk_corona_luminosity(self, w:float, num_integration_points: int = 1000, blackhole_mass=None, r_star=None, inclination_angle=None) -> float:
        """Calculate the luminosity predicted by the ADAF model for this agn in the wavelengths in <w>."""
        freq = c_nm/w
        first_prod = 4*jnp.pi*h_m2kg*jnp.cos(inclination_angle)*freq**3/c_m**2 / (10 * 3.086e+16)**2
        r_seq = jnp.linspace(r_star, 10**4 * r_star, num_integration_points)
        y_seq = slim_disk_corona_integrand_funct(R=r_seq, mass=blackhole_mass, freq=freq)
        second_prod = trapezoid(y_seq, r_seq)
        flux = f_l(f_nu = second_prod * first_prod * 1e3, nu = freq)
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        # flux_density = f_l(f_nu = trapezoid(y_seq, r_seq), nu = freq)
        # slim_disk_luminosity_density = flux_density * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density

    
    @forward
    def luminosity_density(self, w: jnp.ndarray, t: jnp.ndarray, integration_points: int = 1000, perturbations=None) -> jnp.ndarray:
        """ Interpolate the luminosity density ( erg/s/nm ) for a all combinations of the elements in <w> and <t>."""
        trunc_thin_disk_lum = jax.vmap(self.truncated_disk_and_adaf_luminosity_density, in_axes=(0, None))(w, integration_points)
        slim_disk_corona_lum = jax.vmap(self.slim_disk_corona_luminosity, in_axes=(0, None))(w, integration_points)
        # TODO: is this how. to do it?
        condition = self.eddington_normalized_accretion_rate >= m_dot_crit
        base_luminosity = jnp.where(condition, slim_disk_corona_lum,  trunc_thin_disk_lum)
        t = jnp.atleast_1d(t)
        perturbation = jnp.interp(t, self.times, perturbations) # linearly interpolate perturbations
        return base_luminosity[None, :] * perturbation[:, None]
    

class AGNSourceLipunova2018(TransientSource):
    """ Create an AGN source whose SED is modelled according to the standard disk model, as used in lighcurvelynx, based on https://doi.org/10.1007/978-3-319-93009-1_1

    Parameters
    ---------- 
    cosmology: Cosmology. 
        Optional. If given, it can be used to compute the distance modulus from the redshift.
    name: str. 
        Optional. Name of the source.
    blackhole_mass: float. 
        Mass of the black hole in kg
    accretion_rate: float. 
        Accretion rate of the black hole, in kg/year
    inclination_angle: Param(float)
        Inclination angle of the AGN with respect to the observer. The oberver's line of sight makes an angle i 
        to the normal to the disc plane. In radians. 
    r_rin_to_rngin: Param(float)
        Ratio of the inner radius to the gravitational radius of the black hole. 
    times: jnp.ndarray[float] | None
        A vector of length equal to <time_perturbations> whose values equal to the times represented by <time_perturbations>.
    time_perturbations: jnp.ndarray[float] | None
        An array containing all of the values the base luminosities will be multiplied by to cause variability with respect to time. 
        Its length will determine the number of timepoints we will have in the grid. 
    luminosity_density_scaling: float
        A factor by which to divide the luminosity density. This could be used, for example, to correct for bias introduced by the 
        perturbations.
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float 
    accretion_rate: float
    inclination_angle: float
    rin_to_rng: float
    times: jnp.ndarray
    perturbations: jnp.ndarray

    def __init__(self, N, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, accretion_rate: float = None, inclination_angle: float = None, 
                 rin_to_rng: float = 6, real_fourier_perturbations: jnp.ndarray = None, imag_fourier_perturbations: jnp.ndarray = None, start_time: float = None, end_time: float = None, luminosity_density_scaling: float = 1, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="kg")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="kg/year")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.rin_to_rng = Param("rin_to_rng", rin_to_rng, units="dimentionless", description="inner radius of the thin disk")
        self.real_fourier_perturbations = Param("real_fourier_perturbations", real_fourier_perturbations)
        self.imag_fourier_perturbations = Param("imag_fourier_perturbations", imag_fourier_perturbations)
        self.fourier_perturbations = Param("fourier_perturbations", lambda p: p.real_fourier_perturbations.value + 1j * p.imag_fourier_perturbations.value)
        self.fourier_perturbations.link(self.real_fourier_perturbations)
        self.fourier_perturbations.link(self.imag_fourier_perturbations)
        self.N = N
        self.perturbations = Param("perturbations", lambda p: jnp.fft.irfft(p.fourier_perturbations.value, n=self.N), shape=(self.N,), units="dimentionless", description="a vector containing all of the numbers the base luminosities will be " \
        "multiplied by. Cause of the variability with respect to time. Its length will determine the number of time points queried.")
        self.perturbations.link(self.fourier_perturbations)
        self.luminosity_density_scaling = Param("luminosity_density_scaling", luminosity_density_scaling, units="dimentionless", description="a factor by which to divide the luminosity density")
        self.times = jnp.squeeze(jnp.linspace(start_time, end_time, N))
    
    @forward
    def base_luminosity_density(self, w: float | jnp.ndarray, num_integration_points: int = 100, inclination_angle=None, blackhole_mass=None, accretion_rate=None, rin_to_rng=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in erg/s/nm. w must be in nm, and the distance <d> in parsecs."""
        w = jnp.atleast_1d(w)
        freq = c_nm/w
        r_in = rin_to_rng * G * blackhole_mass / c_m**2 # in metres
        r_0 = (7 / 6) ** 2 * r_in # as done in lighcurvelynx from page 31 of https://doi.org/10.1007/978-3-319-93009-1_1
        T_0 = compute_T0(blackhole_mass, accretion_rate, r_in)
        distance = 10 * PARSECS_TO_METRES
        # Set minimum x_in value for numerical stability
        x_in = jnp.maximum(h_m2kg * freq / (k_si * T_0) * (r_in / r_0) ** (3 / 4), 1e-6)
        x_out = x_in + 100
        integral = jax.vmap(quad, in_axes=(None, 0, 0, None))(standard_disk_integration_f, x_in, x_out, num_integration_points)
        flux_freq = 16 * jnp.pi / 3 / distance**2 * jnp.cos(inclination_angle) * (k_si * T_0 / h_m2kg)**(8/3) * h_m2kg * freq**(1/3) * r_0**2 / (c_m**2) * integral * 10**3 # erg/s/cm^2/Hz
        flux = f_l(f_nu=flux_freq, nu =freq)
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density
    
    @forward
    def luminosity_density(self, w: jnp.ndarray, t: jnp.ndarray, integration_points: int = 100, perturbations=None, luminosity_density_scaling=None) -> jnp.ndarray:
        """ Interpolate the luminosity density ( erg/s/nm ) for a all combinations of the elements in <w> and <t>."""
        t = jnp.atleast_1d(t)
        base_luminosity = self.base_luminosity_density(w=w, num_integration_points=integration_points)
        perturbation = jnp.interp(t, self.times, perturbations) # linearly interpolate perturbations
        return base_luminosity[None, :] * jnp.exp(perturbation[:, None]) / luminosity_density_scaling

class AGNSourceLipunova2018_no_fourier(TransientSource):
    """ Create an AGN source whose SED is modelled according to the standard disk model, as used in lighcurvelynx, based on https://doi.org/10.1007/978-3-319-93009-1_1

    Parameters
    ---------- 
    cosmology: Cosmology. 
        Optional. If given, it can be used to compute the distance modulus from the redshift.
    name: str. 
        Optional. Name of the source.
    blackhole_mass: float. 
        Mass of the black hole in kg
    accretion_rate: float. 
        Accretion rate of the black hole, in kg/year
    inclination_angle: Param(float)
        Inclination angle of the AGN with respect to the observer. The oberver's line of sight makes an angle i 
        to the normal to the disc plane. In radians. 
    r_rin_to_rngin: Param(float)
        Ratio of the inner radius to the gravitational radius of the black hole. 
    times: jnp.ndarray[float] | None
        A vector of length equal to <time_perturbations> whose values equal to the times represented by <time_perturbations>.
    time_perturbations: jnp.ndarray[float] | None
        An array containing all of the values the base luminosities will be multiplied by to cause variability with respect to time. 
        Its length will determine the number of timepoints we will have in the grid. 
    luminosity_density_scaling: float
        A factor by which to divide the luminosity density. This could be used, for example, to correct for bias introduced by the 
        perturbations.
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float 
    accretion_rate: float
    inclination_angle: float
    rin_to_rng: float
    times: jnp.ndarray
    perturbations: jnp.ndarray

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, accretion_rate: float = None, inclination_angle: float = None, 
                 rin_to_rng: float = 6, perturbations: jnp.ndarray = None, start_time: float = None, end_time: float = None, luminosity_density_scaling: float = 1, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="kg")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="kg/year")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.rin_to_rng = Param("rin_to_rng", rin_to_rng, units="dimentionless", description="inner radius of the thin disk")
        self.perturbations = Param("perturbations", perturbations)
        self.luminosity_density_scaling = Param("luminosity_density_scaling", luminosity_density_scaling, units="dimentionless", description="a factor by which to divide the luminosity density")
        self.start_time = Param("start_time", start_time)
        self.end_time = Param("end_time", end_time)
    
    @forward
    def base_luminosity_density(self, w: float | jnp.ndarray, num_integration_points: int = 100, inclination_angle=None, blackhole_mass=None, accretion_rate=None, rin_to_rng=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in erg/s/nm. w must be in nm, and the distance <d> in parsecs."""
        w = jnp.atleast_1d(w)
        freq = c_nm/w
        r_in = rin_to_rng * G * blackhole_mass / c_m**2 # in metres
        r_0 = (7 / 6) ** 2 * r_in # as done in lighcurvelynx from page 31 of https://doi.org/10.1007/978-3-319-93009-1_1
        T_0 = compute_T0(blackhole_mass, accretion_rate, r_in)
        distance = 10 * PARSECS_TO_METRES
        # Set minimum x_in value for numerical stability
        x_in = jnp.maximum(h_m2kg * freq / (k_si * T_0) * (r_in / r_0) ** (3 / 4), 1e-6)
        x_out = x_in + 100
        integral = jax.vmap(quad, in_axes=(None, 0, 0, None))(standard_disk_integration_f, x_in, x_out, num_integration_points)
        flux_freq = 16 * jnp.pi / 3 / distance**2 * jnp.cos(inclination_angle) * (k_si * T_0 / h_m2kg)**(8/3) * h_m2kg * freq**(1/3) * r_0**2 / (c_m**2) * integral * 10**3 # erg/s/cm^2/Hz
        flux = f_l(f_nu=flux_freq, nu =freq)
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density
    
    @forward
    def luminosity_density(self, w: jnp.ndarray, t: jnp.ndarray, integration_points: int = 100, perturbations=None, luminosity_density_scaling=None, start_time=None, end_time=None) -> jnp.ndarray:
        """ Interpolate the luminosity density ( erg/s/nm ) for a all combinations of the elements in <w> and <t>."""
        times = jnp.linspace(start_time, end_time, len(perturbations))
        t = jnp.atleast_1d(t)
        base_luminosity = self.base_luminosity_density(w=w, num_integration_points=integration_points)
        perturbation = jnp.interp(t, times, perturbations) # linearly interpolate perturbations
        return base_luminosity[None, :] * jnp.exp(perturbation[:, None]) / luminosity_density_scaling


class AGNSourceLipunova2018_invariable(AGNSourceLipunova2018):
    """ A version of AGNSourceLipunova2018 that does not depend on time, created for fitting purposes. It does not have perturbations. 
    """
    name: str
    cosmology: Cosmology
    blackhole_mass: float 
    accretion_rate: float
    inclination_angle: float
    rin_to_rng: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, accretion_rate: float = None, inclination_angle: float = None, 
                 rin_to_rng: float = 6, luminosity_density_scaling: float = 1, **kwargs) -> None:
        TransientSource.__init__(self, cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                   description="Mass of the black hole", units="kg")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                   description="accretion_rate of the black hole", units="kg/year")
        self.inclination_angle = Param("inclination_angle", inclination_angle, units="radians", description="inclination angle of the AGN with respect to the observer")
        self.rin_to_rng = Param("rin_to_rng", rin_to_rng, units="dimentionless", description="inner radius of the thin disk")
        self.luminosity_density_scaling = Param("luminosity_density_scaling", luminosity_density_scaling, units="dimentionless", description="a factor by which to divide the luminosity density")
    
    @forward
    def luminosity_density(self, w: float | jnp.ndarray, t, blackhole_mass=None, inclination_angle=None, accretion_rate=None, rin_to_rng=None) -> float:
        """ Calculate the time-indepenedent base luminosity density predicted by the thin disk model for this AGN. Return in erg/s/nm. w must be in nm, and the distance <d> in parsecs."""
        w = jnp.atleast_1d(w)
        freq = c_nm/w
        r_in = rin_to_rng * G * blackhole_mass / c_m**2 # in metres
        r_0 = (7 / 6) ** 2 * r_in # as done in lighcurvelynx from page 31 of https://doi.org/10.1007/978-3-319-93009-1_1
        T_0 = compute_T0(blackhole_mass, accretion_rate, r_in)
        distance = 10 * PARSECS_TO_METRES
        # Set minimum x_in value for numerical stability
        x_in = jnp.maximum(h_m2kg * freq / (k_si * T_0) * (r_in / r_0) ** (3 / 4), 1e-6)
        x_out = x_in + 100
        integral = jax.vmap(quad, in_axes=(None, 0, 0, None))(standard_disk_integration_f, x_in, x_out, 100)
        flux_freq = 16 * jnp.pi / 3 / distance**2 * jnp.cos(inclination_angle) * (k_si * T_0 / h_m2kg)**(8/3) * h_m2kg * freq**(1/3) * r_0**2 / (c_m**2) * integral * 10**3 # erg/s/cm^2/Hz
        flux = f_l(f_nu=flux_freq, nu =freq)
        luminosity_density = flux * ABSOLUTE_FLUX_TO_LUM_DENSITY
        return luminosity_density