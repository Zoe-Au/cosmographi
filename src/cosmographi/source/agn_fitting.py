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


def softplus_inverse(x):
    return jnp.log(jnp.exp(x) - 1.0)

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
    def __init__(self, temp=1, name=None, alpha=None, A=None,real_fourier_perturbations=None,  imag_fourier_perturbations=None, **kwargs):
        super().__init__(name)
        self.alpha = ck.Param("alpha", alpha, description="the slope of the power spectrum")
        self.A = ck.Param("A", A, description="the amplitude of the power spectrum")
        self.real_fourier_perturbations = ck.Param("real_fourier_perturbations", real_fourier_perturbations)
        self.imag_fourier_perturbations = ck.Param("imag_fourier_perturbations", imag_fourier_perturbations)
        # self.real_fourier_perturbations = ck.Param("real_fourier_perturbations", lambda p: p.z_r.value*jnp.sqrt(p.P_k.value))
        # self.real_fourier_perturbations.link(self.P_k)
        # self.real_fourier_perturbations.link(self.z_r)
        # self.imag_fourier_perturbations = ck.Param("imag_fourier_perturbations", lambda p: p.z_i.value*jnp.sqrt(p.P_k.value))
        # self.imag_fourier_perturbations.link(self.P_k)
        # self.imag_fourier_perturbations.link(self.z_i)
        self.fourier_perturbations = ck.Param("fourier_perturbations", lambda p: jnp.concatenate([jnp.zeros(1), p.real_fourier_perturbations.value + 1j * p.imag_fourier_perturbations.value]))
        self.fourier_perturbations.link(self.real_fourier_perturbations)
        self.fourier_perturbations.link(self.imag_fourier_perturbations)
        self.perturbations = ck.Param("perturbations", lambda p: jnp.fft.irfft(p.fourier_perturbations.value), units="dimentionless", description="a vector containing all of the numbers the base luminosities will be " \
                "multiplied by. Cause of the variability with respect to time. Its length will determine the number of time points queried.")
        self.perturbations.link(self.fourier_perturbations)
        #TODO: maybe pass this parameter all times instead of deriving it each time
        self.temp = temp
    
    @ck.forward
    def __call__(self, alpha=None, A=None, fourier_perturbations=None, perturbations=None):

        N = perturbations.shape[0]
        # var_floor = 1e-8
        
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
        # add gaussian centred around 0 for the mean
        # flux parameter will absorb changes in average
     
    
#TODO: don't hardcode effects
#TODO: look over  signatures
class LogLikelihoodAGN(ck.Module):
    def __init__(self, y, y_err, start_time, end_time, obs_t, temp=1, name=None, z=None, instrument: Instrument=None, band: jnp.ndarray=None, exp_time: jnp.ndarray=None, sky_brightness: jnp.ndarray=None, PSF_Aeff: jnp.ndarray=None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, imag_fourier_perturbations=None, real_fourier_perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, **kwargs):
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
        c = Cosmology()
        self.source = source_factory(AGNSourceLipunova2018, MWExtinction_Calzetti00)(cosmology=c, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, rin_to_rng=rin_to_rng, inclination_angle=inclination_angle, real_fourier_perturbations=real_fourier_perturbations, imag_fourier_perturbations=imag_fourier_perturbations, start_time=start_time, end_time=end_time,  luminosity_density_scaling=luminosity_density_scaling)
        self.source.z = ck.Param("z", z, description="the redshift of the AGN")
        self.source.t0 = 0 #TODO:??
        self.vobs = jax.vmap(self.instrument.observe, in_axes=(None, 0, 0, 0, 0, None, 0))
        self.temp = temp
        # in the future, source proved by user
        
    @ck.forward
    def __call__(self, key):
        band, exp_time, sky_brightness,PSF_Aeff = jnp.atleast_1d(self.band), jnp.atleast_1d(self.exp_time), jnp.atleast_1d(self.sky_brightness), jnp.atleast_1d(self.PSF_Aeff)
        result = self.vobs(key, band, exp_time, sky_brightness, PSF_Aeff, self.source, self.obs_t)[0] # TODO: is it supposed to be this one? bc there is randomness in the rror too
        return - 0.5* jnp.sum(((self.y - result)/self.y_err)**2) / self.temp # for annealing divide by 100-10-1 acept bad values unstuck, try for prior if it doesnt work, or posterior

class LogPosterior(ck.Module):
    def __init__(self, key, y, y_err, start_time, end_time, obs_t, instrument, band: jnp.ndarray, exp_time: jnp.ndarray, sky_brightness: jnp.ndarray, PSF_Aeff: jnp.ndarray, temp=1, name: str=None, z: float=None, blackhole_mass: float=None, accretion_rate: float=None, inclination_angle: float=None, rin_to_rng: float=None, real_fourier_perturbations=None, imag_fourier_perturbations=None, fourier_perturbations: jnp.ndarray=None, luminosity_density_scaling: float=None, A_V_c00mw: float=None, R_V_c00mw: float=None, alpha: float=None, A: float=None, **kwargs):
        # TODO: leave all of these as parameters?
        super().__init__(name)
        self.log_likelihood = LogLikelihoodAGN(name= "log_likelihood", y=y, y_err=y_err, obs_t=obs_t, temp=temp, end_time=end_time, start_time=start_time, z=z, instrument=instrument, band=band, exp_time=exp_time, sky_brightness=sky_brightness, PSF_Aeff=PSF_Aeff, blackhole_mass=blackhole_mass, accretion_rate=accretion_rate, inclination_angle=inclination_angle, rin_to_rng=rin_to_rng, real_fourier_perturbations=real_fourier_perturbations, imag_fourier_perturbations=imag_fourier_perturbations, luminosity_density_scaling=luminosity_density_scaling, A_V_c00mw=A_V_c00mw, R_V_c00mw=R_V_c00mw) 
        # self.log_prior = PowerLawPrior_logpdf(name="log_prior", alpha=alpha, A=A, temp=temp, z_r=self.log_likelihood.source.z_r, z_i=self.log_likelihood.source.z_i,  P_k=self.log_likelihood.source.P_k, nyquist=self.log_likelihood.source.nyquist)
        self.log_prior = PowerLawPrior_logpdf(name="log_prior", alpha=alpha, A=A, temp=temp, imag_fourier_perturbations=self.log_likelihood.source.imag_fourier_perturbations, real_fourier_perturbations=self.log_likelihood.source.real_fourier_perturbations)
        self.key = key # for annealing
        
    @ck.forward
    def __call__(self, z=None, blackhole_mass=None, accretion_rate=None, inclination_angle=None, rin_to_rng=None, fourier_perturbations=None, luminosity_density_scaling=None, A_V_c00mw=None, R_V_c00mw=None, alpha=None, A=None):
        return (self.log_likelihood(self.key) + self.log_prior())

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
            n_samples = 10
            key, init_key, annealing_key = jax.random.split(self.key, 3)
            init_keys = jax.random.split(init_key, n_chains)
            
            def get_initial_position(base, key):
                # return 0.01 * jax.random.normal(key, shape=jnp.shape(base)) * base + base
                 return base # starting t true value
            
            initial_positions = jax.vmap(get_initial_position, in_axes=(None, 0))(self.log_posterior.get_values(), init_keys)
            warmup = bx.window_adaptation(bx.hmc, j_logposterior, num_integration_steps=1, target_acceptance_rate=0.99, initial_step_size=1e-5) #TODO : tune num_integration_steps and num warmup steps
            print("caling warmup.run")
                    # adaptation_results, adaptation_info = jax.vmap(warmup.run, in_axes=(None, 0))(key, initial_position) 
                    # check step size, check that it is actually accepting samples ~80%
                    # check acceptance rate for real run
                    # annealing 
            adaptation_results, adaptation_info = warmup.run(key, initial_positions[0], num_steps=1000) 
            print('calling warmup complete')
            
            def turn_position_to_state(position, j_logposterior):
                return bx.hmc.init(position, j_logposterior)   
            states = jax.vmap(turn_position_to_state, in_axes=(0, None))(initial_positions, j_logposterior)
            
            @jax.jit
            def one_step(init_states, keys):
                states, info = kernel(keys, init_states)
                print(states)
                return states, (states, info)
                            
            @jax.jit
            def run_chain(init_state, key):
                keys = jax.random.split(key, n_samples)
                return jax.lax.scan(one_step, init_state, keys)
            
            temperatures = [1]
            
             #TODO move whole directory do zoe/zoe (no memory on home)
            
            for temp in temperatures:
                self.log_posterior.log_likelihood.temp = temp
                j_logposterior = jax.jit(self.log_posterior)
                temp_key, annealing_key = jax.random.split(annealing_key)
                chain_keys = jax.random.split(temp_key, n_chains)
                kernel = bx.hmc(j_logposterior, **adaptation_results.parameters).step
                final_states, (sampled_states, infos) = jax.vmap(run_chain)(states, chain_keys)
                states = final_states
                print(temp)
                print(jnp.mean(infos.acceptance_rate))
                print(final_states)
                print("----------------")
                if temp == 1:
                    samples = sampled_states
                d = {}
                for param in j_logposterior.dynamic_params:
                    idx = self.log_posterior.find_index(param)
                    d[param.name] = samples.position[:, :, idx]
                # plot first and last states to see if we are exploring
                return d
    

    def plot(self, true_values: dict[str, any] = None):
        """True values is a dict where param names are keys.
        `array_param` is the one vector-valued parameter; its true values are
        shown in the posterior and corner plots but not the trace plot."""
        d = self.fit()
        idata = az.from_dict(d)
        var_names = list(idata.posterior.data_vars)

        lines = None
        if true_values is not None:
            lines = [(name, {}, true_values[name]) for name in var_names if name in true_values and name != "fourier_perturbations"]
        az.plot_trace(idata, lines=lines)
        plt.show()

        ref_val = None
        if true_values is not None:
            ref_val = []
            for name in var_names:
                if name == "fourier_perturbations" and name in true_values:
                    ref_val.extend(np.asarray(true_values[name]).ravel())
                elif name in true_values:
                    ref_val.append(true_values[name])
                else:
                    ref_val.append(None)
        az.plot_posterior(idata, ref_val=ref_val)
        plt.show()

        pair_idata = idata.copy()
        pair_var_names = []
        pair_reference_values = {}
        for name in var_names:
            if name != "fourier_perturbations":
                pair_var_names.append(name)
                if true_values is not None and name in true_values:
                    pair_reference_values[name] = true_values[name]
            else:
                data = idata.posterior["fourier_perturbations"]
                n_elements = data.sizes["samples"]
                true_array = None
                if true_values is not None and "fourier_perturbations" in true_values:
                    true_array = np.asarray(true_values["fourier_perturbations"]).ravel()
                for i in range(n_elements):
                    scalar_name = f"fourier_perturbations[{i}]"
                    pair_idata.posterior[scalar_name] = (data.isel(samples=i) .drop_vars("samples", errors="ignore"))
                    pair_var_names.append(scalar_name)
                    if true_array is not None:
                        pair_reference_values[scalar_name] = true_array[i]
            az.plot_pair(pair_idata, var_names=pair_var_names, kind="kde", marginals=True,
                reference_values=(pair_reference_values if true_values is not None else None),
                reference_values_kwargs={"color": "red", "marker": "o"})
            plt.show()
        return d


class AGNFitPrior(ck.Module):
    def __init__(self, key: jnp.ndarray, y: jnp.ndarray, y_err: jnp.ndarray, name=None, **kwargs):
        super().__init__(name)
        key, key2 = jax.random.split(key)
        self.log_posterior = LogPosterior(name="log_posterior", key=key, y=y, y_err=y_err, **kwargs)
        self.key = key2

    def fit(self):
        n_chains = 4
        n_samples = 10
        key, init_key, annealing_key = jax.random.split(self.key, 3)
        init_keys = jax.random.split(init_key, n_chains)
        def get_initial_position(base, key):
            # return 0.1 * jax.random.normal(key, shape=jnp.shape(base)) * base + base
            return base # starting t true value
        initial_positions = jax.vmap(get_initial_position, in_axes=(None, 0))(self.log_posterior.get_values(), init_keys)
            # adaptation_results, adaptation_info = jax.vmap(warmup.run, in_axes=(None, 0))(key, initial_position) 
            # check step size, check that it is actually accepting samples ~80%
            # check acceptance rate for real run
            # annealing 
        def turn_position_to_state(position, j_logposterior):
            return bx.hmc.init(position, j_logposterior)   

        #TODO move whole directory do zoe/zoe (no memory on home)
        # temperatures = [1e7, 1e6, 1e5, 1e4, 1e3, 100, 30, 10, 3, 1]
        # temperatures = [1]
        temperatures = [1e7, 1e4, 100, 30, 10, 3, 1]
        # white_noise_levels = [10, 1, 1e-1, 1e-3, 1e-5, 1e-7]

            # for i in range(n_chains):
            #     pos = initial_positions[i]
            #     val = self.log_posterior.log_prior(pos)
            #     grad = jax.grad(self.log_posterior.log_prior)(pos)
            #     print(i, "logdensity:", val)
            #     print(i, "grad norm:", jnp.linalg.norm(grad))
            #     print(i, "grad max abs:", jnp.max(jnp.abs(grad)))
    
        for temp in temperatures:
            self.log_posterior.log_prior.temp = temp
            j_logprior = jax.jit(self.log_posterior.log_prior)
            print(temp)
            warmup = bx.window_adaptation(bx.hmc, j_logprior, num_integration_steps=3, target_acceptance_rate=0.80) #TODO : tune num_integration_steps and num warmup steps
            print("caling warmup.run")
                # adaptation_results, adaptation_info = jax.vmap(warmup.run, in_axes=(None, 0))(key, initial_position) 
                # check step size, check that it is actually accepting samples ~80%
                # check acceptance rate for real run
                # annealing 
            adaptation_results, adaptation_info = warmup.run(key, initial_positions[0])
                # warmup_acceptance_rates = adaptation_info.info.acceptance_rate
                # print("Warmup acceptance rates:")
                # for i, rate in enumerate(warmup_acceptance_rates):
                #     print(f"  warmup step {i + 1}: {float(rate)}")
                # print(f"Mean warmup acceptance rate: "f"{float(jnp.mean(warmup_acceptance_rates))}")
            states = jax.vmap(turn_position_to_state, in_axes=(0, None))(initial_positions, j_logprior)
            temp_key, annealing_key = jax.random.split(annealing_key)
            chain_keys = jax.random.split(temp_key, n_chains)
            kernel = bx.hmc(j_logprior, **adaptation_results.parameters).step

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
        for param in j_logprior.dynamic_params:
            idx = self.log_posterior.find_index(param)
            d[param.name] = samples.position[:, :, idx]
            # plot first and last states to see if we are exploring
        return d



    def plot(self, true_values: dict[str, any] = None):
        d = self.fit()
        idata = az.from_dict(d)

        vector_params = {
            "real_fourier_perturbations",
            "imag_fourier_perturbations",
        }

        var_names = list(idata.posterior.data_vars)

        # ================================================================
        # UNPACK TRUE VALUES ONLY
        #
        # Scalars:
        #   alpha -> {"alpha": 0.7}
        #
        # Vectors:
        #   real_fourier_perturbations ->
        #       {"real_fourier_perturbations_0": ...,
        #        "real_fourier_perturbations_1": ..., ...}
        #
        # We do NOT unpack idata.posterior here.
        # ================================================================

        unpacked_true_values = {}

        if true_values is not None:
            for name, value in true_values.items():

                if name in vector_params:
                    values = np.asarray(value).ravel()

                    for i, v in enumerate(values):
                        unpacked_true_values[f"{name}_{i}"] = float(v)

                else:
                    unpacked_true_values[name] = float(value)

        # ================================================================
        # SEPARATE SCALAR AND VECTOR PARAMETERS
        # ================================================================

        scalar_var_names = [
            name
            for name in var_names
            if name not in vector_params
        ]

        perturbation_var_names = [
            name
            for name in vector_params
            if name in idata.posterior
        ]

        # ================================================================
        # TRACE: SCALARS
        # ================================================================

        scalar_lines = None

        if unpacked_true_values:
            scalar_lines = [
                (
                    name,
                    {},
                    unpacked_true_values[name],
                )
                for name in scalar_var_names
                if name in unpacked_true_values
            ]

        if scalar_var_names:
            az.plot_trace(
                idata,
                var_names=scalar_var_names,
                lines=scalar_lines if scalar_lines else None,
                compact=True,
            )

            plt.tight_layout()
            plt.show()

        # ================================================================
        # TRACE: FOURIER VECTORS
        #
        # No true-value lines here. This avoids constructing one ArviZ
        # line object per Fourier coefficient.
        # ================================================================

        if perturbation_var_names:
            az.plot_trace(
                idata,
                var_names=perturbation_var_names,
                compact=True,
            )

            plt.tight_layout()
            plt.show()

        # ================================================================
        # POSTERIOR: SCALARS
        #
        # IMPORTANT:
        # ArviZ expects:
        #
        # ref_val={
        #     "alpha": [{"ref_val": 0.7}]
        # }
        #
        # rather than:
        #
        # ref_val={"alpha": 0.7}
        # ================================================================

        scalar_ref_val = {}

        for name in scalar_var_names:
            if name in unpacked_true_values:
                scalar_ref_val[name] = [
                    {
                        "ref_val": unpacked_true_values[name]
                    }
                ]

        if scalar_var_names:
            az.plot_posterior(
                idata,
                var_names=scalar_var_names,
                ref_val=(
                    scalar_ref_val
                    if scalar_ref_val
                    else None
                ),
            )

            plt.tight_layout()
            plt.show()

        # ================================================================
        # POSTERIOR: FOURIER VECTORS
        #
        # Don't pass thousands of reference values to ArviZ.
        # This is intentionally reference-line-free for memory safety.
        # ================================================================

        if perturbation_var_names:
            az.plot_posterior(
                idata,
                var_names=perturbation_var_names,
            )

            plt.tight_layout()
            plt.show()

        # ================================================================
        # PAIR PLOT
        #
        # Expand vector posterior variables ONLY for the pair plot.
        # ================================================================

        pair_idata = idata.copy()
        pair_var_names = []
        pair_reference_values = {}

        for name in var_names:

            # ------------------------------------------------------------
            # Scalar
            # ------------------------------------------------------------

            if name not in vector_params:

                pair_var_names.append(name)

                if name in unpacked_true_values:
                    pair_reference_values[name] = (
                        unpacked_true_values[name]
                    )

                continue

            # ------------------------------------------------------------
            # Vector
            # ------------------------------------------------------------

            data = idata.posterior[name]
            dim = f"{name}_dim_0"

            n_elements = data.sizes[dim]

            for i in range(n_elements):

                scalar_name = f"{name}_{i}"

                pair_idata.posterior[scalar_name] = (
                    data.isel({dim: i})
                )

                pair_var_names.append(scalar_name)

                if scalar_name in unpacked_true_values:
                    pair_reference_values[scalar_name] = (
                        unpacked_true_values[scalar_name]
                    )

        # ================================================================
        # PAIR PLOT
        # ================================================================

        if pair_var_names:
            az.plot_pair(
                pair_idata,
                var_names=pair_var_names,
                kind="kde",
                marginals=True,
                reference_values=(
                    pair_reference_values
                    if pair_reference_values
                    else None
                ),
                reference_values_kwargs={
                    "color": "red",
                    "marker": "o",
                },
            )

            plt.tight_layout()
            plt.show()

        return d

