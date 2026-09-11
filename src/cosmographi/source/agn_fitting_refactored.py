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

# add gaussian centred around 0 for the mean
# flux parameter will absorb changes in average
#TODO: don't hardcode effects
#TODO: look over  signatures
# ad priors for all parameters we sample over 
# fix all except perturbations params
# prior over physical parameters of AGN is over population rather than individual source


class Normal_logpdf(ck.Module):
    def __init__(self, name=None, loc=None, scale=None):
        super().__init__(name)
        self.loc = ck.Param("loc", loc, description="the location parameter of the normal distribution")
        self.scale = ck.Param("scale", scale, description="the scale parameter of the normal distribution")

    @ck.forward
    def __call__(self, x, loc=None, scale=None):
        return stats.norm.logpdf(x, loc=loc, scale=scale)

class PowerLawPrior_logpdf(ck.Module):
    def __init__(self, temp=1, name=None, alpha=None, A=None, source=None,**kwargs):
        super().__init__(name)
        self.alpha = ck.Param("alpha", alpha, description="the slope of the power spectrum")
        self.A = ck.Param("A", A, description="the amplitude of the power spectrum")
        self.fourier_perturbations = source.fourier_perturbations
        self.N = source.N
        self.temp = temp
    
    @ck.forward
    def __call__(self, alpha=None, A=None, fourier_perturbations=None, N=None):
        # N = self.source.N
        # 1. Transform to frequency domain
        freqs = jnp.fft.rfftfreq(N)
        
        # 2. Isolate non-zero frequencies
        # The DC component (k=0) is excluded to avoid singular values.
        # It is typically assumed the signal is mean-centered, or the mean
        # is modeled with a separate uniform/normal prior.
        k_nz = freqs[1:]
        x_k_nz = fourier_perturbations[1:]
        
        # 3. Compute expected power spectrum variance
        # Applying the variance floor prevents the variance from dropping to zero
        # (e.g., at high frequencies if alpha < 0) which would cause log(0) NaNs.
        # P_k = jax.nn.softplus(A) / (k_nz ** alpha) 
        P_k = A / (k_nz ** alpha) 
        
        # 4. Calculate log probability of the Fourier modes
        # The rfft coefficients for k > 0 are independent complex Gaussians.
        # The variance of these complex modes is proportional to P(k).
        power_x = jnp.abs(x_k_nz)**2
        
        # Log PDF of a complex Gaussian: -|z|^2 / Var - log(pi * Var)
        log_prob_modes = -(power_x / P_k) - jnp.log(jnp.pi * P_k)

        # Sum over all independent modes to get the total log-prior
        perturbations_prior = jnp.sum(log_prob_modes) # is this ok?
        alpha_prior = Normal_logpdf(name="alpha_prior", loc=0.5, scale=1.0)
        A_prior = Normal_logpdf(name="A_prior", loc=N, scale=N/10)
        return jnp.real((perturbations_prior + alpha_prior(alpha)) + A_prior(A) / self.temp)


class LogLikelihoodAGN(ck.Module):
    def __init__(self, y, y_err, obs_t, temp=1, name=None, z=None, instrument: Instrument=None, band: jnp.ndarray=None, exp_time: jnp.ndarray=None, sky_brightness: jnp.ndarray=None, PSF_Aeff: jnp.ndarray=None, source: AGNSourceLipunova2018=None, **kwargs):
        """times are rest-frame times for which the original source's perturbations are defined
        """
        # TODO: leave all of these as parameters? -> not for band,  exp_time,  sky_brightness, PSF_Aeff
        super().__init__(name)
        self.y = y
        self.y_err = y_err
        self.obs_t = obs_t
        self.instrument = instrument
        self.band = band
        self.exp_time = exp_time
        self.sky_brightness = sky_brightness
        self.PSF_Aeff = PSF_Aeff
        self.source = source
        self.vobs = jax.vmap(self.instrument.observe, in_axes=(None, 0, 0, 0, 0, None, 0))
        self.temp = temp

    @ck.forward
    def __call__(self, key):
        band, exp_time, sky_brightness,PSF_Aeff = jnp.atleast_1d(self.band), jnp.atleast_1d(self.exp_time), jnp.atleast_1d(self.sky_brightness), jnp.atleast_1d(self.PSF_Aeff)
        result = self.vobs(key, band, exp_time, sky_brightness, PSF_Aeff, self.source, self.obs_t)[0] # TODO: is it supposed to be this one? bc there is randomness in the rror too
        return - 0.5* jnp.sum(((self.y - result)/self.y_err)**2) / self.temp # for annealing divide by 100-10-1 acept bad values unstuck, try for prior if it doesnt work, or posterior

class LogPosterior(ck.Module):
    def __init__(self, key, y, y_err, obs_t, instrument, band: jnp.ndarray, exp_time: jnp.ndarray, sky_brightness: jnp.ndarray, PSF_Aeff: jnp.ndarray, temp=1, name=None,source: AGNSourceLipunova2018=None, alpha: float=None, A: float=None, **kwargs):
        # TODO: leave all of these as parameters?
        super().__init__(name)
        self.log_likelihood = LogLikelihoodAGN(name="log_likelihood", y=y, y_err=y_err, obs_t=obs_t, temp=temp, instrument=instrument, band=band, exp_time=exp_time, sky_brightness=sky_brightness, PSF_Aeff=PSF_Aeff, source=source)
        self.log_prior = PowerLawPrior_logpdf(name="log_prior", alpha=alpha, A=A, temp=temp, source=source)
        self.key = key # for annealing
        
    @ck.forward
    def __call__(self, source=None, alpha=None, A=None):
        return (self.log_likelihood(self.key) + self.log_prior())


class AGNFit(ck.Module):
    def __init__(self, key: jnp.ndarray, y: jnp.ndarray, y_err: jnp.ndarray, name=None, **kwargs):
        super().__init__(name)
        key, key2 = jax.random.split(key)
        self.log_posterior = LogPosterior(name="log_posterior", key=key, y=y, y_err=y_err, **kwargs)
        self.key = key2

    def fit(self):
        n_chains = 4
        n_samples = 1000
        key, init_key, annealing_key = jax.random.split(self.key, 3)
        init_keys = jax.random.split(init_key, n_chains)
        def get_initial_position(base, key):
            # return 0.1 * jax.random.normal(key, shape=jnp.shape(base)) * base + base
            return base # starting t true value
        initial_positions = jax.vmap(get_initial_position, in_axes=(None, 0))(self.log_posterior.get_values(), init_keys)
        def turn_position_to_state(position, j_logposterior):
            return bx.hmc.init(position, j_logposterior)   
        temperatures = [1]
        for temp in temperatures:
            self.log_posterior.log_prior.temp = temp
            self.log_posterior.log_likelihood.temp = temp
            j_logpost = jax.jit(self.log_posterior)
            print(temp)
            warmup = bx.window_adaptation(bx.hmc, j_logpost, num_integration_steps=3, target_acceptance_rate=0.80) #TODO : tune num_integration_steps and num warmup steps
            print("caling warmup.run")
            adaptation_results, adaptation_info = warmup.run(key, initial_positions[0])
            states = jax.vmap(turn_position_to_state, in_axes=(0, None))(initial_positions, j_logpost)
            temp_key, annealing_key = jax.random.split(annealing_key)
            chain_keys = jax.random.split(temp_key, n_chains)
            kernel = bx.hmc(j_logpost, **adaptation_results.parameters).step

            @jax.jit
            def one_step(init_states, keys):
                states, info = kernel(keys, init_states)
                print(states)
                return states, (states, info)
                                    
            @jax.jit
            def run_chain(init_state, key):
                keys = jax.random.split(key, n_samples)
                return jax.lax.scan(one_step, init_state, keys)
                
            final_states, (sampled_states, infos) = jax.vmap(run_chain)(states, chain_keys)
            print(infos)
            initial_positions = final_states.position
            print(temp)
            print(jnp.mean(infos.acceptance_rate))
            print(final_states)
            print("----------------")
        samples = sampled_states
        d = {}
        for param in j_logpost.dynamic_params:
            idx = self.log_posterior.find_index(param)
            d[param.name] = samples.position[:, :, idx]
        return d


    def plot(self, true_values: dict[str, any] = None):
        d = self.fit()
        idata = az.from_dict(d)
        vector_params = {"real_fourier_perturbations", "imag_fourier_perturbations"}
        var_names = list(idata.posterior.data_vars)

        unpacked_true_values = {}
        if true_values is not None:
            for name, value in true_values.items():
                if name in vector_params:
                    values = np.asarray(value).ravel()
                    for i, v in enumerate(values):
                        unpacked_true_values[f"{name}_{i}"] = float(v)
                else:
                    unpacked_true_values[name] = float(value)
        scalar_var_names = [name for name in var_names if name not in vector_params]
        perturbation_var_names = [name for name in vector_params if name in idata.posterior]

        scalar_lines = None
        if unpacked_true_values:
            scalar_lines = [(name, {}, unpacked_true_values[name]) for name in scalar_var_names if name in unpacked_true_values]
        if scalar_var_names:
            az.plot_trace(idata, var_names=scalar_var_names, lines=scalar_lines if scalar_lines else None, compact=True)
            plt.tight_layout()
            plt.show()

        if perturbation_var_names:
            az.plot_trace(idata, var_names=perturbation_var_names, compact=True)
            plt.tight_layout()
            plt.show()

        scalar_ref_val = {}
        for name in scalar_var_names:
            if name in unpacked_true_values:
                scalar_ref_val[name] = [{"ref_val": unpacked_true_values[name]}]

        if scalar_var_names:
            az.plot_posterior(idata, var_names=scalar_var_names, ref_val=(scalar_ref_val if scalar_ref_val else None))
            plt.tight_layout()
            plt.show()
        if perturbation_var_names:
            az.plot_posterior(idata, var_names=perturbation_var_names)
            plt.tight_layout()
            plt.show()
            
        pair_idata = idata.copy()
        pair_var_names = []
        pair_reference_values = {}
        for name in var_names:
            if name not in vector_params:
                pair_var_names.append(name)
                if name in unpacked_true_values:
                    pair_reference_values[name] = (unpacked_true_values[name])
                continue
            data = idata.posterior[name]
            dim = f"{name}_dim_0"
            n_elements = data.sizes[dim]
            for i in range(n_elements):
                scalar_name = f"{name}_{i}"
                pair_idata.posterior[scalar_name] = (data.isel({dim: i}))
                pair_var_names.append(scalar_name)
                if scalar_name in unpacked_true_values:
                    pair_reference_values[scalar_name] = (unpacked_true_values[scalar_name])
        if pair_var_names:
            az.plot_pair(pair_idata, var_names=pair_var_names, kind="kde", marginals=True,
                reference_values=(pair_reference_values if pair_reference_values else None),
                reference_values_kwargs={"color": "red", "marker": "o"})
            plt.tight_layout()
            plt.show()
        return d