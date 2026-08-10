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
import numpy as np

class Halfnormal_logpdf(ck.Module):
    def __init__(self, name=None, scale=None):
        super().__init__(name)
        self.scale = ck.Param("scale", scale, description="the scale parameter of the half-normal distribution")

    @ck.forward
    def __call__(self, x, scale=None):
        return jnp.where(x > 0,jnp.log(2.0) + stats.norm.logpdf(x, loc=0.0, scale=scale),- jnp.inf) 

class Normal_logpdf(ck.Module):
    def __init__(self, name=None, loc=None, scale=None):
        super().__init__(name)
        self.loc = ck.Param("loc", loc, description="the location parameter of the normal distribution")
        self.scale = ck.Param("scale", scale, description="the scale parameter of the normal distribution")

    @ck.forward
    def __call__(self, x, loc=None, scale=None):
        return stats.norm.logpdf(x, loc=loc, scale=scale)

class PowerLawPrior_logpdf(ck.Module):
    def __init__(self, name=None, alpha=None, A=None, perturbations=None, **kwargs):
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
        alpha_prior = Normal_logpdf(name="alpha_prior", loc=0.5, scale=1.0)
        A_prior = Halfnormal_logpdf(name="A_prior", scale=1.25)

        return perturbations_prior + alpha_prior(alpha) + A_prior(A)
        # add gaussian centred around 0 for the mean
        # flux parameter will absorb changes in average
     
    
#TODO: don't hardcode effects
#TODO: look over  signatures
class LogLikelihoodAGN(ck.Module):
    def __init__(self, y, y_err, times, name=None, z=None, instrument: Instrument=None, band: jnp.ndarray=None, exp_time: jnp.ndarray=None, sky_brightness: jnp.ndarray=None, PSF_Aeff: jnp.ndarray=None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, **kwargs):
        # TODO: leave all of these as parameters? -> not for band,  exp_time,  sky_brightness, PSF_Aeff
        super().__init__(name)
        self.y = y
        self.y_err = y_err
        self.times = times
        self.instrument = instrument
        self.band = band
        self.exp_time = exp_time
        self.sky_brightness = sky_brightness
        self.PSF_Aeff = PSF_Aeff
        c = Cosmology()
        if times is not None:
            start_time, end_time = times[0], times[-1] 
        else:
            start_time, end_time = None, None
        self.source = source_factory(AGNSourceLipunova2018, MWExtinction_Calzetti00)(cosmology=c, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, rin_to_rng=rin_to_rng, inclination_angle=inclination_angle, perturbations=perturbations, start_time=start_time, end_time=end_time,  luminosity_density_scaling=luminosity_density_scaling)
        self.source.z = ck.Param("z", z, description="the redshift of the AGN")
        self.source.t0 = 0 #TODO:??
        self.vobs = jax.vmap(self.instrument.observe, in_axes=(None, 0, 0, 0, 0, None, 0))
        # in the future, source proved by user
        
    @ck.forward
    def __call__(self, key):
        band, exp_time, sky_brightness,PSF_Aeff = jnp.atleast_1d(self.band), jnp.atleast_1d(self.exp_time), jnp.atleast_1d(self.sky_brightness), jnp.atleast_1d(self.PSF_Aeff)
        result = self.vobs(key, band, exp_time, sky_brightness, PSF_Aeff, self.source, self.times)[0] # TODO: is it supposed to be this one? bc there is randomness in the rror too
        return - 0.5* jnp.sum(((self.y - result)/self.y_err)**2) # for annealing divide by 100-10-1 acept bad values unstuck, try for prior if it doesnt work, or posterior

class LogPosterior(ck.Module):
    def __init__(self, key, y, y_err, times, instrument, band: jnp.ndarray, exp_time: jnp.ndarray, sky_brightness: jnp.ndarray, PSF_Aeff: jnp.ndarray, temp=1, name: str=None, z: float=None, blackhole_mass: float=None, accretion_rate: float=None, inclination_angle: float=None, rin_to_rng: float=None, perturbations: jnp.ndarray=None, luminosity_density_scaling: float=None, A_V_c00mw: float=None, R_V_c00mw: float=None, alpha: float=None, A: float=None, **kwargs):
        # TODO: leave all of these as parameters?
        super().__init__(name)
        self.log_likelihood = LogLikelihoodAGN(name= "log_likelihood", y=y, y_err=y_err, times=times, z=z, instrument=instrument, band=band, exp_time=exp_time, sky_brightness=sky_brightness, PSF_Aeff=PSF_Aeff, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, inclination_angle=inclination_angle, rin_to_rng=rin_to_rng, perturbations=perturbations, luminosity_density_scaling=luminosity_density_scaling, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw) 
        self.log_prior = PowerLawPrior_logpdf(name="log_prior", alpha=alpha, A=A, perturbations=self.log_likelihood.source.perturbations)
        self.key = key
        self.temp = temp # for annealing
        
    @ck.forward
    def __call__(self, z=None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, alpha=None, A=None):
        return (self.log_likelihood(self.key) + self.log_prior()) / self.temp

# ad priors for all parameters we sample over 
# fix all except perturbations params
# prior over physical parameters of AGN is over population rather than individual source

class AGNFit(ck.Module):
    def __init__(self, key: jnp.ndarray, y: jnp.ndarray, y_err: jnp.ndarray, name=None, **kwargs):
        super().__init__(name)
        key, key2 = jax.random.split(key)
        self.log_posterior = LogPosterior(name="log_posterior", key=key, y=y, y_err=y_err, **kwargs)
        self.key = key2

    def fit(self):
        j_logposterior = jax.jit(self.log_posterior) # use everywhere so we compile it only once
        n_chains = 4
        n_samples = 2500
        key, init_key, annealing_key = jax.random.split(self.key, 3)
        init_keys = jax.random.split(init_key, n_chains)

        def get_initial_position(base, key):
            return 0.01 * jax.random.normal(key, shape=jnp.shape(base)) * base + base
            # return base # starting t true value

        initial_positions = jax.vmap(get_initial_position, in_axes=(None, 0))(self.log_posterior.get_values(), init_keys)
        warmup = bx.window_adaptation(bx.hmc, j_logposterior, num_integration_steps=12, target_acceptance_rate=0.8) #TODO : tune num_integration_steps and num warmup steps
        print("caling warmup.run")
        # adaptation_results, adaptation_info = jax.vmap(warmup.run, in_axes=(None, 0))(key, initial_position) 
        # check step size, check that it is actually accepting samples ~80%
        # check acceptance rate for real run
        # annealing 
        adaptation_results, adaptation_info = warmup.run(key, initial_positions[0]) 
        print('calling warmup complete')

        def turn_position_to_state(position, j_logposterior):
                    return bx.hmc.init(position, j_logposterior)
                
        states = jax.vmap(turn_position_to_state, in_axes=(0, None))(initial_positions, j_logposterior)

        @jax.jit
        def one_step(init_states, keys):
            states, _ = kernel(keys, init_states)
            return states, states
                
        @jax.jit
        def run_chain(init_state, key):
            keys = jax.random.split(key, n_samples)
            return jax.lax.scan(one_step, init_state, keys)


        temperatures = [100, 30, 10, 3, 1]

        for temp in temperatures:
            self.log_posterior.log_likelihood.temp = temp
            j_logposterior = jax.jit(self.log_posterior)
            temp_key, annealing_key = jax.random.split(annealing_key)
            chain_keys = jax.random.split(temp_key, n_chains)
            kernel = bx.hmc(j_logposterior, **adaptation_results.parameters).step
            final_states, sampled_states = jax.vmap(run_chain)(states, chain_keys)
            states = final_states
            if temp == 1:
                samples = sampled_states
        d = {}
        for param in j_logposterior.dynamic_params:
            idx = self.log_posterior.find_index(param)
            d[param.name] = samples.position[:, :, idx]
        # plot first and last states to see if we are exploring
        return d

    def plot(self, true_values: dict[str, any]=None):
        """True values is a dict where param names are keys"""
        idata = az.from_dict(self.fit())
        axes = az.plot_trace(idata)
        if true_values is not None:
            for i, var in enumerate(idata.posterior.data_vars):
                name = str(var)
                if name in true_values:
                    value = np.asarray(true_values[name])
                    if value.ndim == 0:
                        axes[i, 0].axvline(value, color="red", linestyle="--")
                        axes[i, 1].axhline(value, color="red", linestyle="--")
        plt.show()
        posterior_axes = az.plot_posterior(idata)
        axes = list(np.asarray(posterior_axes, dtype=object).flat)
        if true_values is not None:
            axis_idx = 0
            for var in idata.posterior.data_vars:
                name = str(var)
                shape = idata.posterior[name].shape[2:]
                n_axes = int(np.prod(shape)) if shape else 1

                if name in true_values:
                    value = np.asarray(true_values[name])
                    if value.ndim == 0:
                        axes[axis_idx].axvline(value, color="red", linestyle="--")
                    else:
                        for j, v in enumerate(value.ravel()):
                            axes[axis_idx + j].axvline(v, color="red", linestyle="--")

                axis_idx += n_axes

        plt.show()

        # Pair plot
        az.plot_pair(idata, kind="kde", marginals=True)

        plt.show()


# class AGNFit(ck.Module):
#     def __init__(self, key, y, y_err, name=None, **kwargs):
#         super().__init__(name)
#         key, key2 = jax.random.split(key)
#         self.log_posterior = LogPosterior(name="log_posterior", key=key, y=y, y_err=y_err, **kwargs)
#         self.key = key2

#     def fit(self):
#         j_logposterior = jax.jit(self.log_posterior) # use everywhere so we compile it only once
#         n_chains = 4
#         n_samples = 5000
#         key, init_key, sample_key = jax.random.split(self.key, 3)
#         init_keys = jax.random.split(init_key, n_chains)

#         def get_initial_position(base, key):
#             return 0.01 * jax.random.normal(key, shape=jnp.shape(base)) * base + base
#             # return base # starting t true value

#         initial_positions = jax.vmap(get_initial_position, in_axes=(None, 0))(self.log_posterior.get_values(), init_keys)
#         warmup = bx.window_adaptation(bx.hmc, j_logposterior, num_integration_steps=12, target_acceptance_rate=0.8) #TODO : tune num_integration_steps and num warmup steps
#         print("caling warmup.run")
#         # adaptation_results, adaptation_info = jax.vmap(warmup.run, in_axes=(None, 0))(key, initial_position) 
#         # check step size, check that it is actually accepting samples ~80%
#         # check acceptance rate for real run
#         # annealing 
#         adaptation_results, adaptation_info = warmup.run(key, initial_positions[0]) 
#         print('calling warmup complete')
#         chain_keys = jax.random.split(sample_key, n_chains)
#         kernel = bx.hmc(jax.jit(j_logposterior), **adaptation_results.parameters).step

#         def turn_position_to_state(position, j_logposterior):
#             return bx.hmc.init(position, j_logposterior)
        
#         init_states = jax.vmap(turn_position_to_state, in_axes=(0, None))(initial_positions, j_logposterior)

#         @jax.jit
#         def one_step(init_states, keys):
#             states, _ = kernel(keys, init_states)
#             return states, states
        
#         @jax.jit
#         def run_chain(init_state, key):
#             keys = jax.random.split(key, n_samples)
#             return jax.lax.scan(one_step, init_state, keys)

#         _, states = jax.vmap(run_chain)(init_states, chain_keys)
#         print("sampling done")
#         d = {}
#         for param in j_logposterior.dynamic_params:
#             idx = self.log_posterior.find_index(param)
#             d[param.name] = states.position[:, :, idx]
#         # plot first and last states to see if we are exploring
#         return d

#     def plot(self, true_values=None):
#         """True values is a dict where param names are keys"""
#         idata = az.from_dict(self.fit())
#         axes = az.plot_trace(idata)
#         if true_values is not None:
#             for i, var in enumerate(idata.posterior.data_vars):
#                 name = str(var)
#                 if name in true_values:
#                     axes[i, 0].axvline(true_values[name], color="red", linestyle="--")
#                     axes[i, 1].axhline(true_values[name], color="red", linestyle="--")
#         plt.show()
#         axes = np.atleast_1d(az.plot_posterior(idata)).ravel()
#         if true_values is not None:
#             for i, var in enumerate(idata.posterior.data_vars):
#                 name = str(var)
#                 if name in true_values:
#                     axes[i].axvline(true_values[name], color="red", linestyle="--")
#         az.plot_pair(idata, kind="kde", marginals=True)

    
        
            