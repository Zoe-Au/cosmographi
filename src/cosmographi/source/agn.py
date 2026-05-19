import os

import eztaox.kernels.quasisep as ekq
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.math_nodes.basic_math_node import BasicMathNode
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.obstable.opsim import OpSim
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.models.eztaox_models import EzTaoXWrapperModel
from lightcurvelynx.utils.plotting import plot_lightcurves
help(simulate_lightcurves)

import jax.numpy as jnp
import jax
from caskade import Param, forward
import numpy as np
from cosmographi.cosmology import Cosmology
from .base import TransientSource
from ..utils import flux
from ..utils.constants import Mpc_to_cm



class AGNSource_Yu2025(TransientSource):
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
    blackhole_mass: float
    accretion_rate: float

    def __init__(self, cosmology: Cosmology = None, name: str = None, blackhole_mass: float = None, 
                 accretion_rate: float = None, **kwargs) -> None:
        super().__init__(cosmology=cosmology, name=name, **kwargs)
        self.blackhole_mass = Param("blackhole_mass", blackhole_mass, shape=(), 
                                    description="Mass of the black hole", units="solar masses")
        self.accretion_rate = Param("accretion_rate", accretion_rate, shape=(), 
                                    description="Accretion rate of the black hole", units="Eddington ratio")    
    
    @forward
    def luminosity_density(self, z: float, w: jnp.ndarray, kernel: ekq.Kernel = ekq.Exp(scale=1, sigma=1), 
                          log_k_params: list|None = None, log_amp_scale: float|None = None, 
                          base_mag: dict[str, float]|None = None) -> jnp.ndarray:
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
        source = self._create_agn_model(z, w, kernel, log_k_params, log_amp_scale, base_mag)
        lightcurves = simulate_lightcurves(source, )


    def _create_agn_model(self, z: float, w: jnp.ndarray, kernel: ekq.Kernel = ekq.Exp(scale=1, sigma=1), 
                          log_k_params: list|None = None, log_amp_scale: float|None = None, 
                          base_mag: dict[str, float]|None = None) -> EzTaoXWrapperModel:
        """
        Return a Gaussian Process model for the AGN lightcurve.

        Precondition: The keys of <base_mag> are strings of the values of <w>.

        #TODO: add more parameters for the GP model, such as the mean function, lag, etc.

        Parameters
        ----------
        kernel: ekq.Kernel|None
            The kernel to use for the Gaussian Process. Default is an exponential kernel.
        k_params: list|None
            Parameters for the kernel. If None, default parameters will be used.
        log_amp_scale: float|None
            log_amp_scale parameters for the Gaussian Process. If None, default parameters will be used. 
        base_mag: dict[str, float]|None
            The base magnitude for the AGN. The keys should be the wavelengths in w as strings.If None, a default value will 
            be used. If 0 is passed, the model output would be the chnage in magnitude.  

        Returns
        -------
        EzTaoXWrapperModel
            The created AGN model.
        """
        # set default kernel parameters if not provided
        if log_k_params is None:
            log_kernel_param = [
                BasicMathNode("log(scale)", scale=NumpyRandomFunc("uniform", low=1.0, high=1000.0, node_label="scale")),
                BasicMathNode("log(sigma)", sigma=NumpyRandomFunc("uniform", low=0.01, high=1.0, node_label="sigma")),]
        # set default GP parameters if not provided
        if log_amp_scale is None:
            log_amp_scale = [0.0, -0.22, -0.69]
        if base_mag is None:
            base_mag = {w[0, col].item(): NumpyRandomFunc("normal", loc = 23.0, scale = 0.5) 
            for col in range(w.shape[1])}
        source = EzTaoXWrapperModel(
            kernel,  
            baseline_mags=base_mag,  
            band_list=[w[0, col].item() for col in range(w.shape[1])],  # A list of band names in order
            log_kernel_param=log_kernel_param,  # A list of setters for the kernel parameters
            log_amp_scale=log_amp_scale,  # A length N list of setters for the amplitude scales per band
            zero_mean=True,
            has_lag=False,
            ra=0.0,
            dec=0.0,
            redshift=z,
            node_label=self.name,  # A node name for convenience
        )
        return source

