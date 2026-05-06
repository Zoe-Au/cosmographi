import cosmographi as cp
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from contextlib import closing
import pandas as pd
from time import time

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # pick whichever GPU has more free memory
np.random.seed(0)
key = jax.random.PRNGKey(0)
colours = ["purple", "green", "red", "orange", "brown", "yellow"]

# Extract survey data from the LSST database
#####################################################################
survey_path = "/home/renee/renee/opsim/baseline_v5.0.1_10yrs.db"
with closing(sqlite3.connect(survey_path)) as conn:
    query = f"""
        SELECT 
            observationId,
            observationStartMJD,
            visitExposureTime,
            numExposures,
            airmass,
            fieldRA, 
            fieldDec, 
            rotTelPos, 
            rotSkyPos, 
            skyBrightness,
            band,
            filter,
            seeingFwhmEff,
            observation_reason
        FROM observations
        LIMIT 200000;
        """
    survey_df = pd.read_sql_query(query, conn)
print(survey_df)
assert (
    survey_df["numExposures"].nunique() == 1
), "All exposures should have the same number of exposures."
survey_t = survey_df["observationStartMJD"].values
survey_ra = survey_df["fieldRA"].values
survey_dec = survey_df["fieldDec"].values
survey_exp_time = survey_df["visitExposureTime"].values
survey_airmass = survey_df["airmass"].values
survey_sky_brightness = survey_df["skyBrightness"].values
survey_seeing = survey_df["seeingFwhmEff"].values
survey_PSF_Aeff = 4 * np.pi * (survey_seeing / 2.355) ** 2  # convert FWHM to effective area
survey_band = survey_df["band"].values
survey_obsreason = survey_df["observation_reason"].values

# Create source
#####################################################################
C = cp.Cosmology()
source_obj = cp.source_factory(cp.SALT2_2021, cp.source.effects.MWExtinction_Calzetti00)
S = source_obj(cosmology=C, A_V_c00mw=0.1, R_V_c00mw=3.1)
S.load_salt2_model()
S.M.to_static()
S.CL.to_static()
print(S)

# Create instrument
#####################################################################
I = cp.RubinObservatory()
#I = cp.RubinObservatory(throughput=cp.Throughput(air_mass=None))
#I = cp.RubinObservatory(throughput=cp.RubinThroughput(air_mass=None))
I.throughput.trim()
I.throughput.to_dynamic() 
survey_iband = cp.utils.bandstr_to_bandidx(I.throughput.bands, survey_band)
nfilter = len(I.throughput.bands)
#I.throughput.air_mass = survey_airmass
print(I)
print(I.get_values().shape)

# Generate mock SN
#####################################################################
Nsn = 300
t_range = (np.min(survey_t), np.max(survey_t))
source_t = np.random.uniform(*t_range, Nsn)
S.t0 = source_t
R = cp.RateConst(cosmology=C, z_min=0.01, z_max=1.0, r=1)
key, subkey = jax.random.split(key)
source_z = R.sample_z(subkey, source_t.shape)
S.z = source_z
print(source_z)
chooseraddec = np.random.choice(
    len(survey_ra), Nsn
)  # fixme, not right distribution, but good enough for testing
source_ra, source_dec = cp.utils.sample_near(
    survey_ra[chooseraddec], survey_dec[chooseraddec], radius_deg=2, n=Nsn
)
print("Matching sources to survey...")
source_matches = cp.utils.cross_match_survey_circle(
    source_tmin=np.array(jax.vmap(lambda z, t: S.min_time({"z": z, "t0": t}))(source_z, source_t)),
    source_tmax=np.array(jax.vmap(lambda z, t: S.max_time({"z": z, "t0": t}))(source_z, source_t)),
    source_ra=source_ra,
    source_dec=source_dec,
    survey_t=survey_t,
    survey_ra=survey_ra,
    survey_dec=survey_dec,
    survey_fov=3.5,  # degrees, LSST field of view
)
print(f"Matched {len(source_matches)} sources to survey observations.")
for i in list(source_matches.keys()):
    # if i != 19:
    #     del source_matches[i]  # fixme remove
    #     continue
    if len(source_matches[i]) < 5:
        del source_matches[
            i
        ]  # drop sources with fewer than 5 matches, not enough data for lightcurve
        continue
print(f"Matched {len(source_matches)} sources worth examining.")
# print(source_matches)
P = cp.source.prior.SALT2_SK16Prior()
x1, c = P.prior_sample(source_t.shape[0])
S.x1 = x1
S.c = c
sel_src = np.sort(list(source_matches.keys()))

print(S)
print(S.get_values().shape)

# Build params input
#####################################################################
source_values = np.array(S.get_values())
survey_values = np.array(I.get_values())
keys = []
bands = []
exp_times = []
sky_brightnesses = []
PSF_Aeffs = []
obs_t = []
obs_reasons = []
source_params = []
survey_params = []
for i in sel_src:
    _subkeys = jax.random.split(key, len(source_matches[i]) + 1)
    key, subkeys = _subkeys[0], _subkeys[1:]
    keys.append(subkeys)
    bands.append(survey_iband[source_matches[i]])
    exp_times.append(survey_exp_time[source_matches[i]])
    sky_brightnesses.append(survey_sky_brightness[source_matches[i]])
    PSF_Aeffs.append(survey_PSF_Aeff[source_matches[i]])
    obs_t.append(survey_t[source_matches[i]])
    obs_reasons.append(survey_obsreason[source_matches[i]])
    source_params.append(jnp.repeat(source_values[i][None, :], len(source_matches[i]), axis=0))
    survey_params.append(survey_airmass[source_matches[i]][:, None])  # shape (n_obs, 1)

keys = jnp.array(np.concatenate(keys))
bands = jnp.array(np.concatenate(bands))
exp_times = jnp.array(np.concatenate(exp_times))
sky_brightnesses = jnp.array(np.concatenate(sky_brightnesses))
PSF_Aeffs = jnp.array(np.concatenate(PSF_Aeffs))
obs_t = jnp.array(np.concatenate(obs_t))
obs_reasons = np.array(np.concatenate(obs_reasons))
source_params = jnp.array(np.concatenate(source_params, axis=0))
survey_params = jnp.array(np.concatenate(survey_params, axis=0))
print("Keys shape:", keys.shape)
print("Bands shape:", bands.shape)
print("Exp times shape:", exp_times.shape)
print("Source params shape:", source_params.shape)
print("Survey params shape:", survey_params.shape)

# Run mock survey lightcurves
#####################################################################
vobs = jax.jit(jax.vmap(I.observe, in_axes=(0, 0, 0, 0, 0, None, 0, 0, 0)), static_argnums=(5,))
start = time()
LCs = []
nobs = len(keys)
batch_size = 10000
for i in range(0, nobs, batch_size):
    LCs.append(
        vobs(
            keys[i : i + batch_size],
            bands[i : i + batch_size],
            exp_times[i : i + batch_size],
            sky_brightnesses[i : i + batch_size],
            PSF_Aeffs[i : i + batch_size],
            S,
            obs_t[i : i + batch_size],
            source_params[i : i + batch_size],
            survey_params[i : i + batch_size],
        )
    )
LCs = tuple(
    jnp.concatenate(tuple(LCs[i][f] for i in range(len(LCs))), axis=0) for f in range(len(LCs[0]))
)
print(LCs[0].block_until_ready().shape)
print(f"Generated lightcurves in {time() - start:.2f} seconds.")
print("LCs shape:", list(lc.shape for lc in LCs))

# Plot results single source
#####################################################################
# print("exposure times:", exp_times)
# print("sky brightnesses:", sky_brightnesses)
# print("PSF effective areas:", PSF_Aeffs)
# print(
#     "source parameters:",
#     source_params,
#     jnp.all(jnp.isclose(source_params, source_params[0], rtol=1e-8, atol=0.0)),
# )
# print("survey observation reasons:", obs_reasons)
seli = 19
spot = sum(len(source_matches[i]) for i in sel_src if i < seli)
ibands = bands[spot : spot + len(source_matches[seli])]
lc_obs = LCs[0][spot : spot + len(source_matches[seli])]
lc_err = LCs[1][spot : spot + len(source_matches[seli])]
lc_err_mag = jnp.abs(
    jax.vmap(I.mag_system.err)(
        lc_obs, lc_err, survey_airmass[spot : spot + len(source_matches[seli])][:, None]
    )
)
lc_obs_mag = jax.vmap(I.mag_system.flux_to_mag)(
    ibands, lc_obs, survey_airmass[spot : spot + len(source_matches[seli])][:, None]
)
lc_true = LCs[2][spot : spot + len(source_matches[seli])]
lc_true_mag = jax.vmap(I.mag_system.flux_to_mag)(
    ibands, lc_true, survey_airmass[spot : spot + len(source_matches[seli])][:, None]
)
lc_err_true = LCs[3][spot : spot + len(source_matches[seli])]
print("(true - obs) / err:", (lc_true - lc_obs) / lc_err)
plt.hist(
    (LCs[2] - LCs[0]) / LCs[1],
    bins=100,
    density=True,
    alpha=0.6,
    color="b",
    label="(true - obs) / err",
)
plt.plot(
    jnp.linspace(-5, 5, 100),
    jax.scipy.stats.norm.pdf(jnp.linspace(-5, 5, 100)),
    color="r",
    linestyle="--",
    label="Standard Normal",
)
plt.xlabel("(true - obs) / err")
plt.title("Distribution of (true - observed) / error for a single source")
plt.legend()
plt.savefig(f"mock_lightcurve_residuals_source_{seli}.pdf", bbox_inches="tight")
plt.show()
plt.hist(
    (LCs[2] - LCs[0]) / LCs[3],
    bins=100,
    density=True,
    alpha=0.6,
    color="b",
    label="(true - obs) / err",
)
plt.plot(
    jnp.linspace(-5, 5, 100),
    jax.scipy.stats.norm.pdf(jnp.linspace(-5, 5, 100)),
    color="r",
    linestyle="--",
    label="Standard Normal",
)
plt.xlabel("(true - obs) / err")
plt.title("Distribution of (true - observed) / error_true for a single source")
plt.legend()
plt.savefig(f"mock_lightcurve_true_residuals_source_{seli}.pdf", bbox_inches="tight")
plt.show()
fig, ax = plt.subplots(figsize=(20, 20))
for iband in np.unique(ibands):
    sel_band = ibands == iband
    ax.errorbar(
        obs_t[spot : spot + len(source_matches[seli])][sel_band],
        lc_obs_mag[sel_band],
        yerr=lc_err_mag[sel_band],
        fmt="o",
        label=f"Observed {I.throughput.bands[iband]}",
        color=colours[iband],
    )
    ax.plot(
        obs_t[spot : spot + len(source_matches[seli])][sel_band],
        lc_true_mag[sel_band],
        color=colours[iband],
        linestyle="--",
        linewidth=1,
        label="True",
    )
# ax.errorbar(
#     obs_t[spot : spot + len(source_matches[seli])],
#     lc_obs,
#     yerr=lc_err,
#     color="r",
#     fmt="o",
#     label="Observed",
# )
# ax.set_title(f"Source {seli}")
ax.text(0.05, 0.95, f"Source {seli}", transform=ax.transAxes, fontsize=8, verticalalignment="top")
ax.set_ylim(18, 30)
ax.invert_yaxis()
# ax.set_xticks([])
# ax.set_yticks([])
spot += len(source_matches[seli])
plt.savefig(f"mock_lightcurve_source_{seli}.pdf", bbox_inches="tight")
plt.show()

# Plot results
#####################################################################
fig, axarr = plt.subplots(10, 10, figsize=(20, 20), sharey=True)
plt.subplots_adjust(hspace=0, wspace=0)
spot = 0
for i, ax in zip(sel_src[:100], axarr.flatten()):
    ibands = bands[spot : spot + len(source_matches[i])]
    lc_obs = LCs[0][spot : spot + len(source_matches[i])]
    lc_err = LCs[1][spot : spot + len(source_matches[i])]
    lc_err_mag = jnp.abs(
        jax.vmap(I.mag_system.err)(
            lc_obs, lc_err, survey_airmass[spot : spot + len(source_matches[i])][:, None]
        )
    )
    lc_obs_mag = jax.vmap(I.mag_system.flux_to_mag)(
        ibands, lc_obs, survey_airmass[spot : spot + len(source_matches[i])][:, None]
    )
    lc_true = LCs[2][spot : spot + len(source_matches[i])]
    lc_true_mag = jax.vmap(I.mag_system.flux_to_mag)(
        ibands, lc_true, survey_airmass[spot : spot + len(source_matches[i])][:, None]
    )
    # lc_err_true = LCs[3][spot : spot + len(source_matches[i])]
    for iband in np.unique(ibands):
        sel_band = ibands == iband
        ax.errorbar(
            obs_t[spot : spot + len(source_matches[i])][sel_band],
            lc_obs_mag[sel_band],
            yerr=lc_err_mag[sel_band],
            fmt="o",
            label=f"Observed {I.throughput.bands[iband]}",
            color=colours[iband],
        )
        ax.plot(
            obs_t[spot : spot + len(source_matches[i])][sel_band],
            lc_true_mag[sel_band],
            color=colours[iband],
            linestyle="--",
            linewidth=1,
            label="True",
        )
    # ax.errorbar(
    #     obs_t[spot : spot + len(source_matches[i])],
    #     lc_obs,
    #     yerr=lc_err,
    #     color="r",
    #     fmt="o",
    #     label="Observed",
    # )
    # ax.set_title(f"Source {i}")
    ax.text(0.05, 0.95, f"Source {i}", transform=ax.transAxes, fontsize=8, verticalalignment="top")
    ax.set_ylim(18, 30)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    spot += len(source_matches[i])
plt.savefig("mock_lightcurves.pdf", bbox_inches="tight")
plt.show()
