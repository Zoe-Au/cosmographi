import caskade as ck
import jax.numpy as jnp
import jax.scipy.stats as stats
from ..instrument import Instrument
from ..cosmology import Cosmology
from .agn import AGNSourceLipunova2018
from .. import source_factory
from .effects import MWExtinction_Calzetti00
import jax
import blackjax as bx
import matplotlib.pyplot as plt
import arviz as az
import time

class Halfnormal_logpdf(ck.Module):
    def __init__(self, name = None, scale = None, **kwargs):
        super().__init__(name)
        self.scale = ck.Param("scale", scale, description="the scale parameter of the half-normal distribution")

    @ck.forward
    def __call__(self, x, scale = None):
        return jnp.where(x > 0,jnp.log(2.0) + stats.norm.logpdf(x, loc=0.0, scale=scale),- jnp.inf) 

class Normal_logpdf(ck.Module):
    def __init__(self, name = None, loc = None, scale = None, **kwargs):
        super().__init__(name)
        self.loc = ck.Param("loc", loc, description="the location parameter of the normal distribution")
        self.scale = ck.Param("scale", scale, description="the scale parameter of the normal distribution")

    @ck.forward
    def __call__(self, x, loc = None, scale = None):
        return stats.norm.logpdf(x, loc=loc, scale=scale)

class PowerLawPrior_logpdf(ck.Module):
    def __init__(self, name = None, alpha = None, A = None, perturbations = None, **kwargs):
        super().__init__(name)
        self.alpha = ck.Param("alpha", alpha, description="the slope of the power spectrum")
        self.A = ck.Param("A", A, description="the amplitude of the power spectrum")
        self.perturbations = perturbations
    
    @ck.forward
    def __call__(self, alpha=None, A=None, perturbations=None):
        var_floor = 1e-8
        N = perturbations.shape[0]
        
        # 1. Transform to frequency domain
        x_k = jnp.fft.rfft(perturbations)
        freqs = jnp.fft.rfftfreq(N)
        
        # 2. Isolate non-zero frequencies
        # The DC component (k=0) is excluded to avoid singular values.
        # It is typically assumed the signal is mean-centered, or the mean
        # is modeled with a separate uniform/normal prior.
        k_nz = freqs[1:]
        x_k_nz = x_k[1:]
        
        # 3. Compute expected power spectrum variance
        # Applying the variance floor prevents the variance from dropping to zero
        # (e.g., at high frequencies if alpha < 0) which would cause log(0) NaNs.
        P_k = A / (k_nz ** alpha) + var_floor # changed to match generate_perturbations_vector
        
        # 4. Calculate log probability of the Fourier modes
        # The rfft coefficients for k > 0 are independent complex Gaussians.
        # The variance of these complex modes is proportional to P(k).
        power_x = jnp.abs(x_k_nz)**2
        
        # Log PDF of a complex Gaussian: -|z|^2 / Var - log(pi * Var)
        log_prob_modes = -(power_x / P_k) - jnp.log(jnp.pi * P_k)
        
        # Sum over all independent modes to get the total log-prior
        perturbations_prior = jnp.sum(log_prob_modes) - jnp.mean(perturbations)**2
        alpha_prior = Normal_logpdf(name="alpha_prior", loc=1.7, scale=5.0)
        A_prior = Halfnormal_logpdf(name="A_prior", scale=10.0)

        return perturbations_prior + alpha_prior(alpha) + A_prior(A)
        # add gaussian centred around 0 for the mean
        # flux parameter will absorb changes in average
     
    
#TODO: don't hardcode effects
class LogLikelihoodAGN(ck.Module):
    def __init__(self, name = None, y = None, y_err = None, times = None, z = None, instrument: Instrument = None, band: jnp.ndarray = None, exp_time: jnp.ndarray = None, sky_brightness: jnp.ndarray = None, PSF_Aeff: jnp.ndarray = None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, **kwargs):
        # TODO: leave all of these as parameters?
        super().__init__(name)
        self.y = y
        self.y_err = y_err
        self.times = times
        self.instrument = instrument
        self.band = ck.Param("band", band, description="the filter indexes")
        self.exp_time = ck.Param("exp_time", exp_time, description="the exposure times for each observation")
        self.sky_brightness = ck.Param("sky_brightness", sky_brightness, description="the sky brightness for each observation")
        self.PSF_Aeff = ck.Param("PSF_Aeff", PSF_Aeff, description="the effective area of the PSF for each observation")   
        c = Cosmology()
        if times is not None:
            start_time, end_time = times[0], times[-1] 
        else:
            start_time, end_time = None, None
        self.source = source_factory(AGNSourceLipunova2018, MWExtinction_Calzetti00)(cosmology=c, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, rin_to_rng=rin_to_rng, inclination_angle=inclination_angle, perturbations=perturbations, start_time=start_time, end_time=end_time,  luminosity_density_scaling=luminosity_density_scaling)
        self.source.z = ck.Param("z", z, description="the redshift of the AGN")
        self.source.t0 = 0 #TODO:??
        self.vobs = jax.vmap(self.instrument.observe, in_axes=(None, 0, 0, 0, 0, None, 0))
        
    @ck.forward
    def __call__(self, key, band: jnp.ndarray = None, exp_time: jnp.ndarray = None, sky_brightness: jnp.ndarray = None, PSF_Aeff: jnp.ndarray = None):
        band, exp_time, sky_brightness,PSF_Aeff = jnp.atleast_1d(band), jnp.atleast_1d(exp_time), jnp.atleast_1d(sky_brightness), jnp.atleast_1d(PSF_Aeff)
        result = self.vobs(key, band, exp_time, sky_brightness, PSF_Aeff, self.source, self.times)[0] # TODO: is it supposed to be this one? bc there is randomness in the rror too
        return - 0.5* jnp.sum(((self.y - result)/self.y_err)**2)

class LogPosterior(ck.Module):
    def __init__(self, key, instrument, name = None, y = None, y_err = None, times = None, z = None, band: jnp.ndarray = None, exp_time: jnp.ndarray = None, sky_brightness: jnp.ndarray = None, PSF_Aeff: jnp.ndarray = None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, alpha=None, A=None, **kwargs):
        # TODO: leave all of these as parameters?
        super().__init__(name)
        self.log_likelihood = LogLikelihoodAGN(name= "log_likelihood", y=y, y_err=y_err, times=times, z=z, instrument=instrument, band=band, exp_time=exp_time, sky_brightness=sky_brightness, PSF_Aeff=PSF_Aeff, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, inclination_angle=inclination_angle, rin_to_rng=rin_to_rng, perturbations=perturbations, luminosity_density_scaling=luminosity_density_scaling, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw) 
        self.log_prior = PowerLawPrior_logpdf(name="log_prior", alpha=alpha, A=A, perturbations=self.log_likelihood.source.perturbations)
        self.key = key
        
    @ck.forward
    def __call__(self, times = None, z = None, band: jnp.ndarray = None, exp_time: jnp.ndarray = None, sky_brightness: jnp.ndarray = None, PSF_Aeff: jnp.ndarray = None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, alpha=None, A=None):
        return self.log_likelihood(self.key) + self.log_prior()

#TODO: be able to 
class AGNFit(ck.Module):
    def __init__(self, key, name=None, y=None, y_err=None, **kwargs):
        super().__init__(name)
        key, key2 = jax.random.split(key)
        self.log_posterior = LogPosterior(name="log_posterior", key=key, y=y, y_err=y_err, **kwargs)
        self.key = key2

    def fit(self):
        key, key2 = jax.random.split(self.key)
        initial_position = self.log_posterior.get_values()
        print(initial_position)
        lp = self.log_posterior(initial_position)
        print(lp)
        t0 = time.time()
        jax.grad(self.log_posterior)(initial_position)
        print(time.time() - t0)
        warmup = bx.window_adaptation(bx.hmc, self.log_posterior, num_integration_steps=15)
        print("caling warmup.run")
        adaptation_results, adaptation_info = warmup.run(key, initial_position)
        print('calling warmup complete')
        N = 10000
        keys = jax.random.split(key2, N)
        kernel = bx.hmc(self.log_posterior, **adaptation_results.parameters).step
        def one_step(state, key):
            state, info = kernel(key, state)
            return state, state
        @jax.jit
        def run_chain(initial_state, keys):
            return jax.lax.scan(one_step, initial_state, keys)
        final_state, history = run_chain(adaptation_results.state, keys)
        print("sampling done")
        d = {}
        for param in self.log_posterior.dynamic_params:
            idx = self.log_posterior.find_index(param)
            d[param.name] = history.position[:, idx]
        return d

    def plot(self, true_values=None):
        """True values is a dict where param names are keys"""
        idata = az.from_dict(self.fit())
        axes = az.plot_trace(idata)
        if true_values is not None:
            for i, var in enumerate(idata.posterior.data_vars):
                name = str(var)
                if name in true_values:
                    axes[i, 0].axhline(true_values[name], color="red", linestyle="--")
                    axes[i, 1].axvline(true_values[name], color="red", linestyle="--")
        plt.show()
        az.plot_posterior(idata)
        az.plot_pair(idata, kind="kde", marginals=True)

    
        
            