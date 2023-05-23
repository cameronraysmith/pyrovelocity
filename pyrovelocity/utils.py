import functools
import logging
import pdb
from inspect import getmembers
from pprint import pprint
from types import FunctionType
from typing import Tuple

import anndata
import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
import torch
from pytorch_lightning.utilities import rank_zero_only
from scipy.sparse import issparse
from scvi.data import synthetic_iid
from sklearn.decomposition import PCA
from termcolor import colored
from torch.nn.functional import relu
from torch.nn.functional import softmax
from torch.nn.functional import softplus


def trace(func):
    def wrapper(*args, **kwargs):
        params = list(args) + [f"{k}={v}" for k, v in kwargs.items()]
        params_str = ",\n    ".join(map(str, params))
        print(f"{func.__name__}(\n    {params_str},\n)")
        return func(*args, **kwargs)

    return wrapper


def inv(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the element-wise reciprocal of a tensor.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with element-wise reciprocal of x.

    Examples:
        >>> import torch
        >>> x = torch.tensor([2., 4., 0.5])
        >>> inv(x)
        tensor([0.5000, 0.2500, 2.0000])
    """
    return x.reciprocal()


def log(x):
    """
    Computes the element-wise natural logarithm of a tensor, while clipping
    its values to avoid numerical instability.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with element-wise natural logarithm of x.

    Examples:
        >>> import torch
        >>> x = torch.tensor([0.0001, 0.5, 0.9999])
        >>> log(x)
        tensor([-9.2103e+00, -6.9315e-01, -1.0002e-04])
    """
    eps = torch.finfo(x.dtype).eps
    return torch.log(x.clamp(eps, 1 - eps))


def velocity_dus_dt(alpha, beta, gamma, tau, x):
    """
    Computes the velocity du/dt and ds/dt.

    Args:
        alpha (torch.Tensor): Alpha parameter.
        beta (torch.Tensor): Beta parameter.
        gamma (torch.Tensor): Gamma parameter.
        tau (torch.Tensor): Time points.
        x (Tuple[torch.Tensor, torch.Tensor]): Tuple containing u and s.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing du/dt and ds/dt.

    Examples:
        >>> import torch
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> tau = torch.tensor(2.0)
        >>> x = (torch.tensor(1.0), torch.tensor(0.5))
        >>> velocity_dus_dt(alpha, beta, gamma, tau, x)
        (tensor(0.1000), tensor(0.2500))
    """
    u, s = x
    du_dt = alpha - beta * u
    ds_dt = beta * u - gamma * s
    return du_dt, ds_dt


def rescale_time(dx_dt, t_start, t_end):
    """
    Converts an ODE to be solved on a batch of different time intervals into an
    equivalent system of ODEs to be solved on [0, 1].

    Args:
        dx_dt (Callable): Function representing the ODE system.
        t_start (torch.Tensor): Start time of the time interval.
        t_end (torch.Tensor): End time of the time interval.

    Returns:
        Callable: Function representing the rescaled ODE system.

    Examples:
        >>> import torch
        >>> def dx_dt(t, x):
        ...     return -x
        >>> t_start = torch.tensor(0.0)
        >>> t_end = torch.tensor(1.0)
        >>> rescaled_dx_dt = rescale_time(dx_dt, t_start, t_end)
        >>> rescaled_dx_dt(torch.tensor(0.5), torch.tensor(2.0))
        tensor(-2.)
    """
    dt = t_end - t_start

    @functools.wraps(dx_dt)
    def dx_ds(s, *args):
        # move any batch dimensions in s to the left of all batch dimensions in dt
        # s_shape = (1,) if len(s.shape) == 0 else s.shape  # XXX unnecessary?
        s = s.reshape(s.shape + (1,) * len(dt.shape))
        t = s * dt + t_start
        xts = dx_dt(t, *args)
        if isinstance(xts, tuple):
            xss = tuple(xt * dt for xt in xts)
        else:
            xss = xts * dt
        return xss

    return dx_ds


def ode_mRNA(tau, u0, s0, alpha, beta, gamma):
    """
    Solves the ODE system for mRNA dynamics.

    Args:
        tau (torch.Tensor): Time points.
        u0 (torch.Tensor): Initial value of u.
        s0 (torch.Tensor): Initial value of s.
        alpha (torch.Tensor): Alpha parameter.
        beta (torch.Tensor): Beta parameter.
        gamma (torch.Tensor): Gamma parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the final values of u and s.
    """
    """
    Examples:
        >>> import torch
        >>> tau = torch.tensor(2.0)
        >>> u0 = torch.tensor(1.0)
        >>> s0 = torch.tensor(0.5)
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> ode_mRNA(tau, u0, s0, alpha, beta, gamma)
        (tensor(0.6703), tensor(0.4596))
    """
    dx_dt = functools.partial(velocity_dus_dt, alpha, beta, gamma)
    dx_dt = rescale_time(
        dx_dt, torch.tensor(0.0, dtype=tau.dtype, device=tau.device), tau
    )
    grid = torch.linspace(0.0, 1.0, 100, dtype=tau.dtype, device=tau.device)
    x0 = (u0.expand_as(tau), s0.expand_as(tau))
    # xts = odeint_adjoint(dx_dt, x0, grid, adjoint_params=(alpha, beta, gamma, tau))
    xts = odeint(dx_dt, x0, grid)
    uts, sts = xts
    ut, st = uts[-1], sts[-1]
    return ut, st


def mRNA(
    tau: torch.Tensor,
    u0: torch.Tensor,
    s0: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mRNA dynamics given the parameters and initial conditions.

    Args:
        tau (torch.Tensor): Time points.
        u0 (torch.Tensor): Initial value of u.
        s0 (torch.Tensor): Initial value of s.
        alpha (torch.Tensor): Alpha parameter.
        beta (torch.Tensor): Beta parameter.
        gamma (torch.Tensor): Gamma parameter.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the final values of u and s.

    Examples:
        >>> import torch
        >>> tau = torch.tensor(2.0)
        >>> u0 = torch.tensor(1.0)
        >>> s0 = torch.tensor(0.5)
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> mRNA(tau, u0, s0, alpha, beta, gamma)
        (tensor(1.1377), tensor(0.9269))
    """
    expu, exps = torch.exp(-beta * tau), torch.exp(-gamma * tau)

    # invalid values caused by below codes:
    # gamma equals beta will raise inf, inf * 0 leads to nan
    expus = (alpha - u0 * beta) * inv(gamma - beta) * (exps - expu)
    # solution 1: conditional zero filling
    # solution 1 issue:AutoDelta map_estimate of alpha,beta,gamma,switching will become nan, thus u_inf/s_inf/ut/st all lead to nan
    # expus = torch.where(torch.isclose(gamma, beta), expus.new_zeros(1), expus)

    ut = u0 * expu + alpha / beta * (1 - expu)
    st = (
        s0 * exps + alpha / gamma * (1 - exps) + expus
    )  # remove expus is the most stable, does it theoretically make sense?

    # solution 2: conditional analytical solution
    # solution 2 issue:AutoDelta map_estimate of alpha,beta,gamma,switching will become nan, thus u_inf/s_inf/ut/st all lead to nan
    # On the Mathematics of RNA Velocity I: Theoretical Analysis: Equation (2.12) when gamma == beta
    st2 = s0 * expu + alpha / beta * (1 - expu) - (alpha - beta * u0) * tau * expu
    ##st2 = s0 * expu + alpha / gamma * (1 - expu) - (alpha - gamma * u0) * tau * expu
    st = torch.where(torch.isclose(gamma, beta), st2, st)

    # solution 3: do not use AutoDelta and map_estimate? customize guide function?
    # use solution 3 with st2
    return ut, st


def tau_inv(u=None, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None):
    """
    Computes the inverse tau given the parameters and initial conditions.

    Args:
        u (torch.Tensor): Value of u.
        s (torch.Tensor): Value of s.
        u0 (torch.Tensor): Initial value of u.
        s0 (torch.Tensor): Initial value of s.
        alpha (torch.Tensor): Alpha parameter.
        beta (torch.Tensor): Beta parameter.
        gamma (torch.Tensor): Gamma parameter.

    Returns:
        torch.Tensor: Inverse tau.

    Examples:
        >>> import torch
        >>> u = torch.tensor(0.6703)
        >>> s = torch.tensor(0.4596)
        >>> u0 = torch.tensor(1.0)
        >>> s0 = torch.tensor(0.5)
        >>> alpha = torch.tensor(0.5)
        >>> beta = torch.tensor(0.4)
        >>> gamma = torch.tensor(0.3)
        >>> tau_inv(u, s, u0, s0, alpha, beta, gamma)
        tensor(3.9736e-07)
    """
    beta_ = beta * inv(gamma - beta)
    xinf = alpha / gamma - beta_ * (alpha / beta)
    tau1 = -1.0 / gamma * log((s - beta_ * u - xinf) * inv(s0 - beta_ * u0 - xinf))

    uinf = alpha / beta
    tau2 = -1 / beta * log((u - uinf) * inv(u0 - uinf))
    tau = torch.where(beta > gamma, tau1, tau2)
    return relu(tau)


def mse_loss_sum(u_model, s_model, u_data, s_data):
    """
    Computes the mean squared error loss sum between the model and data.

    Args:
        u_model (torch.Tensor): Predicted values of u from the model.
        s_model (torch.Tensor): Predicted values of s from the model.
        u_data (torch.Tensor): True values of u from the data.
        s_data (torch.Tensor): True values of s from the data.

    Returns:
        torch.Tensor: Mean squared error loss sum.

    Examples:
        >>> import torch
        >>> u_model = torch.tensor([0.5, 0.6])
        >>> s_model = torch.tensor([0.7, 0.8])
        >>> u_data = torch.tensor([0.4, 0.5])
        >>> s_data = torch.tensor([0.6, 0.7])
        >>> mse_loss_sum(u_model, s_model, u_data, s_data)
        tensor(0.0200)
    """
    return ((u_model - u_data) ** 2 + (s_model - s_data) ** 2).mean(0)


def get_velocity_samples(posterior_samples, model):
    """
    Computes the velocity samples from the posterior samples.

    Args:
        posterior_samples (dict): Dictionary containing posterior samples.
        model: Model used for predictions.

    Returns:
        torch.Tensor: Velocity samples.

    Examples:
        >>> import torch
        >>> posterior_samples = {
        ...     "beta": torch.tensor([[0.4]]),
        ...     "gamma": torch.tensor([[0.3]]),
        ...     "u_scale": torch.tensor([[[1.0, 2.0]]]),
        ...     "s_scale": torch.tensor([[[2.0, 4.0]]]),
        ...     "u": torch.tensor([[1.0, 2.0]]),
        ...     "s": torch.tensor([[0.5, 1.0]])
        ... }
        >>> model = None  # Model is not used in the function
        >>> get_velocity_samples(posterior_samples, model)
        tensor([[0.6500, 1.3000]])
    """
    beta = posterior_samples["beta"].mean(0)[0]
    gamma = posterior_samples["gamma"].mean(0)[0]
    scale = (
        posterior_samples["u_scale"][:, 0, :] / posterior_samples["s_scale"][:, 0, :]
    ).mean(0)
    ut = posterior_samples["u"] / scale
    st = posterior_samples["s"]
    v = beta * ut - gamma * st
    return v


def mae(pred_counts, true_counts):
    """
    Computes the mean average error between predicted counts and true counts.

    Args:
        pred_counts (np.ndarray): Predicted counts.
        true_counts (np.ndarray): True counts.

    Returns:
        float: Mean average error.

    Examples:
        >>> import numpy as np
        >>> pred_counts = np.array([[1, 2], [3, 4]])
        >>> true_counts = np.array([[2, 3], [4, 5]])
        >>> mae(pred_counts, true_counts)
        1.0
    """
    error = np.abs(true_counts - pred_counts).sum()
    total = pred_counts.shape[0] * pred_counts.shape[1]
    return (error / total).mean().item()


def init_with_all_cells(
    adata,
    input_type="knn",
    shared_time=True,
    latent_factor="linear",
    latent_factor_size=10,
    plate_size=2,
    num_aux_cells=200,
    init_smooth=True,
):
    ## hard to use unsmoothed data for initialization
    ## always initialize the model with smoothed data
    if "Mu" in adata.layers and "Ms" in adata.layers and init_smooth:
        u_obs = torch.tensor(adata.layers["Mu"], dtype=torch.float32)
        s_obs = torch.tensor(adata.layers["Ms"], dtype=torch.float32)
    elif "spliced" in adata.layers:
        u_obs = torch.tensor(
            adata.layers["unspliced"].toarray()
            if issparse(adata.layers["unspliced"])
            else adata.layers["unspliced"],
            dtype=torch.float32,
        )
        s_obs = torch.tensor(
            adata.layers["spliced"].toarray()
            if issparse(adata.layers["spliced"])
            else adata.layers["spliced"],
            dtype=torch.float32,
        )
    else:
        raise

    ub_u = torch.stack(
        [torch.quantile(u[u > 0], 0.99) for u in torch.unbind(u_obs, axis=1)]
    )
    ub_s = torch.stack(
        [torch.quantile(s[s > 0], 0.99) for s in torch.unbind(s_obs, axis=1)]
    )
    s_mask = (s_obs > 0) & (s_obs <= ub_s)
    u_mask = (u_obs > 0) & (u_obs <= ub_u)
    # include zeros
    training_mask = s_mask & u_mask

    u_scale = torch.stack(
        [u[u > 0].std() for u in torch.unbind(u_obs * training_mask, axis=1)]
    )
    s_scale = torch.stack(
        [s[s > 0].std() for s in torch.unbind(s_obs * training_mask, axis=1)]
    )
    scale = u_scale / s_scale

    lb_steady_u = torch.stack(
        [
            torch.quantile(u[u > 0], 0.98)
            for u in torch.unbind(u_obs * training_mask, axis=1)
        ]
    )
    lb_steady_s = torch.stack(
        [
            torch.quantile(s[s > 0], 0.98)
            for s in torch.unbind(s_obs * training_mask, axis=1)
        ]
    )

    steady_u_mask = training_mask & (u_obs >= lb_steady_u)
    steady_s_mask = training_mask & (s_obs >= lb_steady_s)

    u_obs = u_obs / scale
    u_inf = (u_obs * (steady_u_mask | steady_s_mask)).sum(dim=0) / (
        steady_u_mask | steady_s_mask
    ).sum(dim=0)
    s_inf = (s_obs * steady_s_mask).sum(dim=0) / steady_s_mask.sum(dim=0)

    gamma = (u_obs * steady_s_mask * s_obs).sum(axis=0) / (
        (steady_s_mask * s_obs) ** 2
    ).sum(axis=0) + 1e-6
    gamma = torch.where(gamma < 0.05 / scale, gamma * 1.2, gamma)
    gamma = torch.where(gamma > 1.5 / scale, gamma / 1.2, gamma)
    alpha = gamma * s_inf
    beta = alpha / u_inf

    switching = tau_inv(u_inf, s_inf, 0.0, 0.0, alpha, beta, gamma)
    tau = tau_inv(u_obs, s_obs, 0.0, 0.0, alpha, beta, gamma)
    tau = torch.where(tau >= switching, switching, tau)
    tau_ = tau_inv(u_obs, s_obs, u_inf, s_inf, 0.0, beta, gamma)
    tau_ = torch.where(
        tau_ >= tau_[s_obs > 0].max(dim=0)[0], tau_[s_obs > 0].max(dim=0)[0], tau_
    )
    ut, st = mRNA(tau, 0.0, 0.0, alpha, beta, gamma)
    ut_, st_ = mRNA(tau_, u_inf, s_inf, 0.0, beta, gamma)

    u_scale_ = u_scale / scale
    state_on = ((ut - u_obs) / u_scale_) ** 2 + ((st - s_obs) / s_scale) ** 2
    state_off = ((ut_ - u_obs) / u_scale_) ** 2 + ((st_ - s_obs) / s_scale) ** 2
    cell_gene_state_logits = state_on - state_off
    cell_gene_state = cell_gene_state_logits < 0
    t = torch.where(cell_gene_state_logits < 0, tau, tau_ + switching)
    init_values = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "switching": switching,
        "latent_time": t,
        "u_scale": u_scale,
        "s_scale": s_scale,
        "u_inf": u_inf,
        "s_inf": s_inf,
    }
    if input_type == "knn":
        init_values["mask"] = training_mask

    init_values["cell_gene_state"] = cell_gene_state.int()
    if latent_factor == "linear":
        if "spliced" in adata.layers:
            u_obs = torch.tensor(
                adata.layers["unspliced"].toarray()
                if issparse(adata.layers["unspliced"])
                else adata.layers["unspliced"],
                dtype=torch.float32,
            )
            s_obs = torch.tensor(
                adata.layers["spliced"].toarray()
                if issparse(adata.layers["spliced"])
                else adata.layers["spliced"],
                dtype=torch.float32,
            )
        test = np.hstack([u_obs, s_obs])
        pca = PCA(n_components=latent_factor_size)
        pca.fit(test)
        X_train_pca = pca.transform(test)
        init_values["cell_codebook"] = torch.tensor(pca.components_)
        init_values["u_pcs_mean"] = torch.tensor(pca.mean_[: adata.shape[1]])
        init_values["s_pcs_mean"] = torch.tensor(pca.mean_[adata.shape[1] :])
        if plate_size == 2:
            init_values["cell_code"] = (
                torch.tensor(X_train_pca).unsqueeze(-1).transpose(-1, -2)
            )
        else:
            init_values["cell_code"] = torch.tensor(X_train_pca)

    if num_aux_cells > 0:
        np.random.seed(99)
        if "cytotrace" in adata.obs.columns:
            order_aux = np.array_split(np.sort(adata.obs["cytotrace"].values), 50)
            order_aux_list = []
            for i in order_aux:
                order_aux_list.append(
                    np.random.choice(
                        np.where(
                            (adata.obs["cytotrace"].values > i[0])
                            & (adata.obs["cytotrace"].values < i[-1])
                        )[0]
                    )
                )
            order_aux = np.array(order_aux_list)
        else:
            order_aux = np.argsort((adata.layers["spliced"].toarray() > 0).sum(axis=1))[
                ::-1
            ]

        init_values["order_aux"] = torch.from_numpy(order_aux[:num_aux_cells].copy())

        if input_type == "raw":
            u_obs = adata.layers["raw_unspliced"].toarray()
            s_obs = adata.layers["raw_spliced"].toarray()
        init_values["aux_u_obs"] = torch.tensor(u_obs[order_aux][:num_aux_cells])
        init_values["aux_s_obs"] = torch.tensor(s_obs[order_aux][:num_aux_cells])
        init_values["cell_gene_state_aux"] = cell_gene_state[:num_aux_cells]
        init_values["latent_time_aux"] = t[:num_aux_cells]

        if latent_factor == "linear":
            init_values["cell_code_aux"] = (
                torch.tensor(X_train_pca)
                .unsqueeze(-1)
                .transpose(-1, -2)[:num_aux_cells]
            )

    if shared_time:
        cell_time = t.mean(dim=-1, keepdims=True)
        init_values["cell_time"] = cell_time
        init_values["latent_time"] = init_values["latent_time"] - cell_time

        if num_aux_cells > 0:
            init_values["cell_time_aux"] = cell_time[:num_aux_cells]
            init_values["latent_time_aux"] = (
                init_values["latent_time_aux"] - init_values["cell_time_aux"]
            )

    for key in init_values:
        print(key, init_values[key].shape)
        print(init_values[key].isnan().sum())
        assert init_values[key].isnan().sum() == 0
    return init_values


def ensure_numpy_array(obj):
    return obj.toarray() if hasattr(obj, "toarray") else obj


def mae_evaluate(posterior_samples, adata):
    maes_list = []
    labels = []
    if isinstance(posterior_samples, tuple):
        for model_label, model_obj, split_index in zip(
            ["Poisson train", "Poisson valid"],
            posterior_samples[:2],
            posterior_samples[2:],
        ):
            for sample in range(model_obj["u"].shape[0]):
                maes_list.append(
                    mae(
                        np.hstack([model_obj["u"][sample], model_obj["s"][sample]]),
                        np.hstack(
                            [
                                adata.layers["raw_unspliced"].toarray()[split_index],
                                adata.layers["raw_spliced"].toarray()[split_index],
                            ]
                        ),
                    )
                )
                labels.append(model_label)
    else:
        for model_label, model_obj in zip(["Poisson all cells"], [posterior_samples]):
            for sample in range(model_obj["u"].shape[0]):
                maes_list.append(
                    mae(
                        np.hstack([model_obj["u"][sample], model_obj["s"][sample]]),
                        np.hstack(
                            [
                                ensure_numpy_array(adata.layers["raw_unspliced"]),
                                ensure_numpy_array(adata.layers["raw_spliced"]),
                            ]
                        ),
                    )
                )
                labels.append(model_label)

    import pandas as pd

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 4)
    df = pd.DataFrame({"MAE": maes_list, "label": labels})
    sns.boxplot(x="label", y="MAE", data=df, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    print(df.groupby("label").mean())
    return df


def debug(x):
    if torch.any(torch.isnan(x)):
        print("nan number: ", torch.isnan(x).sum())
        pdb.set_trace()


def site_is_discrete(site: dict) -> bool:
    return (
        site["type"] == "sample"
        and not site["is_observed"]
        and getattr(site["fn"], "has_enumerate_support", False)
    )


def get_pylogger(name=__name__, log_level="DEBUG") -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)s: %(message)s"
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        # datefmt=None,
        # reset=True,
        # log_colors={
        #     "debug": "cyan",
        #     "info": "green",
        #     "warning": "yellow",
        #     "error": "red",
        #     "exception": "red",
        #     "fatal": "red",
        #     "critical": "red",
        #     },
    )

    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(name)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def attributes(obj):
    """
    get object attributes
    """
    disallowed_names = {
        name for name, value in getmembers(type(obj)) if isinstance(value, FunctionType)
    }
    return {
        name: getattr(obj, name)
        for name in dir(obj)
        if name[0] != "_" and name not in disallowed_names and hasattr(obj, name)
    }


def print_attributes(obj):
    """
    print object attributes
    """
    pprint(attributes(obj))


def pretty_print_dict(d: dict):
    for key, value in d.items():
        key_colored = colored(key, "green")
        value_lines = str(value).split("\n")
        value_colored = "\n".join(colored(line, "white") for line in value_lines)
        print(f"{key_colored}:\n{value_colored}\n")


def filter_startswith_dict(dictionary_with_underscore_keys):
    """Remove entries from a dictionary whose keys start with an underscore.

    Args:
        dictionary_with_underscore_keys (dict): Dictionary to be filtered.

    Returns:
        dict: Filtered dictionary.
    """
    return {
        k: v
        for k, v in dictionary_with_underscore_keys.items()
        if not k.startswith("_")
    }


def print_anndata(anndata_obj):
    """
    Print a formatted representation of an AnnData object.

    This function produces a custom output for the AnnData object with each
    element of obs, var, uns, obsm, varm, layers, obsp, varp indented and
    displayed on a new line.

    Args:
        anndata_obj (anndata.AnnData): The AnnData object to be printed.

    Raises:
        AssertionError: If the input object is not an instance of anndata.AnnData.

    Examples:
        >>> import anndata
        >>> import numpy as np
        >>> import pandas as pd
        >>> np.random.seed(42)
        >>> X = np.random.randn(10, 5)
        >>> obs = pd.DataFrame({"clusters_coarse": np.random.randint(0, 2, 10),
        ...                     "clusters": np.random.randint(0, 2, 10),
        ...                     "S_score": np.random.rand(10),
        ...                     "G2M_score": np.random.rand(10)})
        >>> var = pd.DataFrame({"gene_name": [f"gene_{i}" for i in range(5)]})
        >>> adata = anndata.AnnData(X, obs=obs, var=var)
        >>> print_anndata(adata)  # doctest: +NORMALIZE_WHITESPACE
        AnnData object with n_obs × n_vars = 10 × 5
            obs:
                clusters_coarse,
                clusters,
                S_score,
                G2M_score,
            var:
                gene_name,
    """
    assert isinstance(
        anndata_obj, anndata.AnnData
    ), "Input object must be of type AnnData."

    def format_elements(elements):
        formatted = "\n".join([f"        {elem}," for elem in elements])
        return formatted

    print(
        f"AnnData object with n_obs × n_vars = {anndata_obj.n_obs} × {anndata_obj.n_vars}"
    )

    properties = {
        "obs": anndata_obj.obs.columns,
        "var": anndata_obj.var.columns,
        "uns": anndata_obj.uns.keys(),
        "obsm": anndata_obj.obsm.keys(),
        "varm": anndata_obj.varm.keys(),
        "layers": anndata_obj.layers.keys(),
        "obsp": anndata_obj.obsp.keys(),
        "varp": anndata_obj.varp.keys(),
    }

    for prop_name, elements in properties.items():
        if len(elements) > 0:
            print(f"    {prop_name}:\n{format_elements(elements)}")


def generate_sample_data(
    n_obs: int = 100,
    n_vars: int = 12,
    alpha: float = 5,
    beta: float = 0.5,
    gamma: float = 0.3,
    alpha_: float = 0,
    noise_model: str = "gillespie",
    random_seed: int = 0,
) -> anndata.AnnData:
    """
    Generate synthetic single-cell RNA sequencing data with spliced and unspliced layers.
    If using the "iid" noise model, the data will be generated with scvi.data.synthetic_iid.
    If using the "normal" or "gillespie" noise model, the data will be generated with
    scvelo.datasets.simulation accounting for the given expression dynamics parameters.

    Args:
        n_obs (int, optional): Number of observations (cells). Default is 100.
        n_vars (int, optional): Number of variables (genes). Default is 12.
        alpha (float, optional): Transcription rate. Default is 5.
        beta (float, optional): Splicing rate. Default is 0.5.
        gamma (float, optional): Degradation rate. Default is 0.3.
        alpha_ (float, optional): Additional transcription rate. Default is 0.
        noise_model (str, optional): Noise model to be used. Must be one of 'iid', 'gillespie', or 'normal'. Default is 'gillespie'.
        random_seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        anndata.AnnData: An AnnData object containing the generated synthetic data.

    Raises:
        ValueError: If noise_model is not one of 'iid', 'gillespie', or 'normal'.

    Examples:
        >>> from pyrovelocity.utils import generate_sample_data, print_anndata
        >>> adata = generate_sample_data(random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(n_obs=50, n_vars=10, noise_model="normal", random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(n_obs=50, n_vars=10, noise_model="iid", random_seed=99)
        >>> print_anndata(adata)
        >>> adata = generate_sample_data(noise_model="wishful thinking")
        Traceback (most recent call last):
            ...
        ValueError: noise_model must be one of 'iid', 'gillespie', 'normal'
    """
    if noise_model == "iid":
        adata = synthetic_iid(
            batch_size=n_obs,
            n_genes=n_vars,
            n_batches=1,
            n_labels=1,
        )
        adata.layers["spliced"] = adata.X.copy()
        adata.layers["unspliced"] = adata.X.copy()
    elif noise_model in {"gillespie", "normal"}:
        adata = scv.datasets.simulation(
            random_seed=random_seed,
            n_obs=n_obs,
            n_vars=n_vars,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            alpha_=alpha_,
            noise_model=noise_model,
        )
    else:
        raise ValueError("noise_model must be one of 'iid', 'gillespie', 'normal'")
    return adata


def anndata_counts_to_df(adata):
    spliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_spliced"]),
        index=list(adata.obs_names),
        columns=list(adata.var_names),
    )
    unspliced_df = pd.DataFrame(
        ensure_numpy_array(adata.layers["raw_unspliced"]),
        index=list(adata.obs_names),
        columns=list(adata.var_names),
    )

    spliced_melted = spliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="spliced"
    )
    unspliced_melted = unspliced_df.reset_index().melt(
        id_vars="index", var_name="var_name", value_name="unspliced"
    )

    df = spliced_melted.merge(unspliced_melted, on=["index", "var_name"])

    df = df.rename(columns={"index": "obs_name"})

    total_obs = adata.n_obs
    total_var = adata.n_vars

    max_spliced = adata.layers["raw_spliced"].max()
    max_unspliced = adata.layers["raw_unspliced"].max()

    return (
        df,
        total_obs,
        total_var,
        max_spliced,
        max_unspliced,
    )
