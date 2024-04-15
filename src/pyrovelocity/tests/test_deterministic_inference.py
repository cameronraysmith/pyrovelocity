import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arviz import InferenceData
from beartype import beartype
from beartype.typing import Tuple
from numpyro.infer import MCMC
from numpyro.infer import NUTS
from numpyro.infer import Predictive
from returns.result import Success
from xarray import Dataset

from pyrovelocity.logging import configure_logging
from pyrovelocity.models import (
    deterministic_transcription_splicing_probabilistic_model,
)
from pyrovelocity.models._deterministic_inference import (
    generate_posterior_inference_data,
)
from pyrovelocity.models._deterministic_inference import (
    generate_prior_inference_data,
)
from pyrovelocity.models._deterministic_inference import (
    generate_test_data_for_deterministic_model_inference,
)
from pyrovelocity.models._deterministic_inference import (
    print_inference_data_structure,
)
from pyrovelocity.models._deterministic_inference import save_inference_plots


logger = configure_logging(__name__)


@pytest.fixture
def setup_observational_data():
    return generate_test_data_for_deterministic_model_inference(
        num_genes=1,
        num_cells=3,
        num_timepoints=4,
        num_modalities=2,
    )


@beartype
@pytest.fixture
def setup_prior_inference_data(
    setup_observational_data
) -> Tuple[
    InferenceData,
    int,
    int,
]:
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data

    num_chains = 1
    num_samples = 10

    idata = generate_prior_inference_data(
        times=times,
        data=data,
        num_chains=num_chains,
        num_samples=num_samples,
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        num_modalities=num_modalities,
    )

    return idata, num_chains, num_samples


@beartype
@pytest.fixture
def setup_posterior_inference_data(
    setup_observational_data
) -> Tuple[
    InferenceData,
    int,
    int,
    int,
]:
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data

    num_chains = 1
    num_samples = 10
    num_warmup = 5

    idata_posterior = generate_posterior_inference_data(
        times=times,
        data=data,
        num_chains=num_chains,
        num_samples=num_samples,
        num_genes=num_genes,
        num_cells=num_cells,
        num_timepoints=num_timepoints,
        num_modalities=num_modalities,
        num_warmup=num_warmup,
    )

    return idata_posterior, num_chains, num_samples, num_warmup


def test_priors(setup_observational_data):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data
    rng_key = jax.random.PRNGKey(0)

    def model():
        deterministic_transcription_splicing_probabilistic_model(times, data)

    predictive = Predictive(model, num_samples=500)
    prior_samples = predictive(rng_key)

    assert "initial_conditions" in prior_samples
    assert "gamma" in prior_samples
    assert "sigma" in prior_samples

    u0_samples, s0_samples = (
        prior_samples["initial_conditions"][:, :, 0],
        prior_samples["initial_conditions"][:, :, 1],
    )
    assert u0_samples.shape == (500, num_genes)
    assert s0_samples.shape == (500, num_genes)
    assert jnp.all(
        u0_samples > 0
    ), "All initial u0 values should be positive (LogNormal)"
    assert jnp.all(
        s0_samples > 0
    ), "All initial s0 values should be positive (LogNormal)"

    gamma_samples = prior_samples["gamma"]
    assert gamma_samples.shape == (500, num_genes)
    assert jnp.all(
        gamma_samples > 0
    ), "All gamma values should be positive (LogNormal)"

    sigma_samples = prior_samples["sigma"]
    assert sigma_samples.shape == (500, num_modalities)
    assert jnp.all(
        sigma_samples > 0
    ), "All sigma values should be positive (HalfNormal)"


def test_ode_solution(setup_observational_data):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data
    rng_key = jax.random.PRNGKey(0)

    def model():
        deterministic_transcription_splicing_probabilistic_model(times, data)

    predictive = Predictive(model, num_samples=10)
    samples = predictive(rng_key)

    observations = samples["observations"]
    assert observations.shape == (
        10,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    )
    assert jnp.all(
        jnp.isfinite(observations)
    ), "Observations contain non-finite values"

    u_pred = observations[..., 0]
    s_pred = observations[..., 1]
    assert u_pred.shape == (10, num_genes, num_cells, num_timepoints)
    assert s_pred.shape == (10, num_genes, num_cells, num_timepoints)
    assert jnp.all(
        jnp.isfinite(u_pred)
    ), "u predictions contain non-finite values"
    assert jnp.all(
        jnp.isfinite(s_pred)
    ), "s predictions contain non-finite values"


def test_model_sampling_statements_prior_predictive(setup_observational_data):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    prior_predictive = Predictive(
        deterministic_transcription_splicing_probabilistic_model,
        num_samples=7,
    )
    prior_predictions = prior_predictive(rng_key_, times=times, data=data)

    assert (
        "observations" in prior_predictions
    ), "Predictive samples should include 'observations'."
    assert prior_predictions["observations"].shape == (
        7,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ), "Shape of prior predictive observations incorrect."


def test_model_sampling_statements_posterior_predictive(
    setup_observational_data
):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data
    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_key_ = jax.random.split(rng_key)

    nuts_kernel = NUTS(deterministic_transcription_splicing_probabilistic_model)
    mcmc = MCMC(nuts_kernel, num_warmup=50, num_samples=50, num_chains=1)
    mcmc.run(rng_key_, times=times, data=data)

    samples = mcmc.get_samples()
    posterior_predictive = Predictive(
        deterministic_transcription_splicing_probabilistic_model,
        samples,
    )
    posterior_predictions = posterior_predictive(
        rng_key_, times=times, data=data
    )

    assert (
        "observations" in posterior_predictions
    ), "Posterior predictive samples should include 'observations'."
    assert posterior_predictions["observations"].shape == (
        50,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ), "Shape of posterior predictive observations incorrect."


def test_generate_prior_inference_data(
    setup_observational_data,
    setup_prior_inference_data,
):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data

    (
        idata_prior,
        num_chains,
        num_samples,
    ) = setup_prior_inference_data

    assert isinstance(
        idata_prior, az.InferenceData
    ), "Output should be an ArviZ InferenceData object"

    assert (
        idata_prior.prior.coords["cells"].size == num_cells
    ), "Cell coordinate should match num_cells"

    assert (
        idata_prior.prior.coords["genes"].size == num_genes
    ), "Gene coordinate should match num_genes"

    assert (
        idata_prior.prior.coords["timepoints"].size == num_timepoints
    ), "Timepoint coordinate should match num_timepoints"

    assert (
        "observations" in idata_prior.prior
    ), "Observations variable should be part of the prior group"

    assert idata_prior.prior["observations"].shape == (
        num_chains,
        num_samples,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ), "Shape of observations should match"

    assert idata_prior.prior["initial_conditions"].shape == (
        num_chains,
        num_samples,
        num_genes,
        num_modalities,
    ), "Shape of initial_conditions should match"

    assert idata_prior.prior["gamma"].shape == (
        num_chains,
        num_samples,
        num_genes,
    ), "Shape of gamma should match"

    assert idata_prior.prior["sigma"].shape == (
        num_chains,
        num_samples,
        num_modalities,
    ), "Shape of sigma should match"


def test_generate_posterior_inference_data(
    setup_observational_data,
    setup_posterior_inference_data,
):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data

    (
        idata_posterior,
        num_chains,
        num_samples,
        num_warmup,
    ) = setup_posterior_inference_data

    assert isinstance(
        idata_posterior, az.InferenceData
    ), "Output should be an ArviZ InferenceData object"

    assert (
        "chain" in idata_posterior.posterior.coords
    ), "Chain dimension should be present"
    assert (
        "draw" in idata_posterior.posterior.coords
    ), "Draw dimension should be present"
    assert (
        idata_posterior.posterior.coords["chain"].size == num_chains
    ), "Chain coordinate should match num_chains"
    assert (
        idata_posterior.posterior.coords["draw"].size == num_samples
    ), "Draw coordinate should match num_samples"

    assert (
        "observations" in idata_posterior.posterior_predictive
    ), "Posterior predictive should contain observations"
    assert idata_posterior.posterior_predictive["observations"].shape == (
        num_chains,
        num_samples,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ), "Shape of posterior predictive observations incorrect"

    assert (
        "initial_conditions" in idata_posterior.prior
    ), "Initial conditions should be in prior"
    assert "gamma" in idata_posterior.prior, "Gamma should be in prior"
    assert "sigma" in idata_posterior.prior, "Sigma should be in prior"

    assert idata_posterior.prior["initial_conditions"].shape == (
        num_chains,
        num_samples,
        num_genes,
        num_modalities,
    ), "Shape of initial_conditions should be correct"

    assert idata_posterior.prior["gamma"].shape == (
        num_chains,
        num_samples,
        num_genes,
    ), "Shape of gamma should be correct"

    assert idata_posterior.prior["sigma"].shape == (
        num_chains,
        num_samples,
        num_modalities,
    ), "Shape of sigma should be correct"


def test_generate_inference_data_plots(
    setup_observational_data,
    setup_prior_inference_data,
    setup_posterior_inference_data,
    tmp_path,
):
    (
        times,
        data,
        num_genes,
        num_cells,
        num_timepoints,
        num_modalities,
    ) = setup_observational_data

    (
        idata_prior,
        num_chains,
        num_samples,
    ) = setup_prior_inference_data

    (
        idata_posterior,
        num_chains,
        num_samples,
        num_warmup,
    ) = setup_posterior_inference_data

    data.shape
    idata_prior.prior["observations"].shape
    idata_posterior.posterior_predictive["observations"].shape

    output_dir = tmp_path / "output_plots"
    logger.info(
        f"\nTest inference data plots will be saved to:\n" f"{output_dir}\n\n"
    )

    result = save_inference_plots(
        idata_prior=idata_prior,
        idata_posterior=idata_posterior,
        output_dir=output_dir,
    )

    assert isinstance(result, Success), "The result should be a Success type."
    assert result.unwrap() is True, "The result should unwrap to True."

    expected_files = [
        "prior_predictive_checks.png",
        "prior_predictive_checks.pdf",
        "posterior_predictive_checks.png",
        "posterior_predictive_checks.pdf",
        "prior_initial_conditions.png",
        "prior_initial_conditions.pdf",
        "posterior_initial_conditions.png",
        "posterior_initial_conditions.pdf",
        "forest_initial_conditions.png",
        "forest_initial_conditions.pdf",
        "prior_gamma.png",
        "prior_gamma.pdf",
        "posterior_gamma.png",
        "posterior_gamma.pdf",
        "forest_gamma.png",
        "forest_gamma.pdf",
        "prior_sigma.png",
        "prior_sigma.pdf",
        "posterior_sigma.png",
        "posterior_sigma.pdf",
        "forest_sigma.png",
        "forest_sigma.pdf",
        "trace_plots.png",
        "trace_plots.pdf",
    ]

    for filename in expected_files:
        file_path = output_dir / filename
        assert file_path.exists(), f"File {filename} does not exist."


@pytest.fixture
def simple_inference_data():
    """Create a simple InferenceData object for testing."""
    data = np.random.rand(10, 3)
    dataset = Dataset({"parameter": (("draw", "chain"), data)})
    return az.InferenceData(posterior=dataset)


def test_print_inference_data_structure(simple_inference_data):
    """Test that the structure description is correct for a simple case."""
    expected_output = (
        "Overview of InferenceData structure:\n"
        "\nGroup: posterior\n"
        "  Variables and their dimensions:\n"
        "  parameter: (draw=10, chain=3)\n"
    )
    actual_output = print_inference_data_structure(simple_inference_data)
    assert (
        actual_output == expected_output
    ), "Output structure description did not match expected."


def test_empty_inference_data():
    """Test that the function handles an empty InferenceData object gracefully."""
    empty_idata = az.InferenceData()
    expected_output = "Overview of InferenceData structure:\n"
    actual_output = print_inference_data_structure(empty_idata)
    assert (
        actual_output == expected_output
    ), "Output should be gracefully handled for empty InferenceData."
