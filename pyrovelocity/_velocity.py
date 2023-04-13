import logging
import os
import pickle
import sys
from statistics import harmonic_mean
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import mlflow
import numpy as np
import pyro
import scvi
import torch
from anndata import AnnData
from numpy import ndarray
from scipy import sparse
from scvi.data import AnnDataManager
from scvi.data._constants import _MODEL_NAME_KEY
from scvi.data._constants import _SCVI_UUID_KEY
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.data._constants import _SETUP_METHOD_NAME
from scvi.data.fields import LayerField
from scvi.data.fields import NumericalObsField
from scvi.model._utils import parse_use_gpu_arg
from scvi.model.base import BaseModelClass
from scvi.model.base._utils import _initialize_model
from scvi.model.base._utils import _load_saved_files
from scvi.model.base._utils import _validate_var_names
from scvi.module.base import PyroBaseModuleClass

from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import vector_field_uncertainty

from ._trainer import VelocityTrainingMixin
from ._velocity_module import VelocityModule
from .utils import init_with_all_cells


# from pyrovelocity.data import setup_anndata_multilayers


logger = logging.getLogger(__name__)


class PyroVelocity(VelocityTrainingMixin, BaseModelClass):
    def __init__(
        self,
        adata: AnnData,
        input_type: str = "raw",
        shared_time: bool = True,
        model_type: str = "auto",
        guide_type: str = "auto",
        likelihood: str = "Poisson",
        t_scale_on: bool = False,
        plate_size: int = 2,
        latent_factor: str = "none",
        latent_factor_operation: str = "selection",
        inducing_point_size: int = 0,
        latent_factor_size: int = 0,
        include_prior: bool = False,
        use_gpu: Union[bool, int] = 0,
        init: bool = False,
        num_aux_cells: int = 0,
        only_cell_times: bool = True,
        decoder_on: bool = False,
        add_offset: bool = False,
        correct_library_size: Union[bool, str] = True,
        cell_specific_kinetics: Optional[str] = None,
        kinetics_num: Optional[int] = None,
    ) -> None:
        """
        PyroVelocity class for estimating RNA velocity and related tasks.

        Args:
            adata (AnnData): An AnnData object containing the gene expression data.
            input_type (str, optional): Type of input data. Can be "raw", "knn", or "raw_cpm". Defaults to "raw".
            shared_time (bool, optional): Whether to use shared time. Defaults to True.
            model_type (str, optional): Type of model to use. Defaults to "auto".
            guide_type (str, optional): Type of guide to use. Defaults to "auto".
            likelihood (str, optional): Type of likelihood to use. Defaults to "Poisson".
            t_scale_on (bool, optional): Whether to use t_scale. Defaults to False.
            plate_size (int, optional): Size of the plate. Defaults to 2.
            latent_factor (str, optional): Type of latent factor. Defaults to "none".
            latent_factor_operation (str, optional): Operation to perform on the latent factor. Defaults to "selection".
            inducing_point_size (int, optional): Size of inducing points. Defaults to 0.
            latent_factor_size (int, optional): Size of latent factors. Defaults to 0.
            include_prior (bool, optional): Whether to include prior information. Defaults to False.
            use_gpu (Union[bool, int], optional): Whether and which GPU to use. Defaults to 0. Can be False.
            init (bool, optional): Whether to initialize the model. Defaults to False.
            num_aux_cells (int, optional): Number of auxiliary cells. Defaults to 0.
            only_cell_times (bool, optional): Whether to use only cell times. Defaults to True.
            decoder_on (bool, optional): Whether to use decoder. Defaults to False.
            add_offset (bool, optional): Whether to add offset. Defaults to False.
            correct_library_size (Union[bool, str], optional): Whether to correct library size or method to correct. Defaults to True.
            cell_specific_kinetics (Optional[str], optional): Type of cell-specific kinetics. Defaults to None.
            kinetics_num (Optional[int], optional): Number of kinetics. Defaults to None.

        Examples:
            >>> import numpy as np
            >>> import anndata
            >>> from pyrovelocity._velocity import PyroVelocity
            >>> from pyrovelocity.utils import pretty_print_dict, print_anndata, generate_sample_data
            >>> n_obs = 3000
            >>> n_vars = 500
            >>> adata = generate_sample_data(n_obs=n_obs, n_vars=n_vars)
            >>> print_anndata(adata)
            >>> adata.layers['raw_spliced'] = adata.layers['spliced']
            >>> adata.layers['raw_unspliced'] = adata.layers['unspliced']
            >>> adata.obs['u_lib_size_raw'] = adata.layers['raw_unspliced'].sum(-1)
            >>> adata.obs['s_lib_size_raw'] = adata.layers['raw_spliced'].sum(-1)
            >>> print_anndata(adata)
            >>> print(adata.X)
            >>> print(adata.layers['spliced'])
            >>> print(adata.layers['unspliced'])
            >>> print(adata.obs['u_lib_size_raw'])
            >>> print(adata.obs['s_lib_size_raw'])
            >>> PyroVelocity.setup_anndata(adata)
            >>> model = PyroVelocity(adata)
            >>> model.train_faster(max_epochs=5, use_gpu=0)
            >>> model.save_model('pyrovel_test_saved_model', overwrite=True)
            >>> model = PyroVelocity.load_model('pyrovel_test_saved_model', adata, use_gpu=0)
            >>> posterior_samples = model.posterior_samples(model.adata, num_samples=30)
            >>> pretty_print_dict(posterior_samples)
            >>> print(posterior_samples.keys())
            >>> model = PyroVelocity(adata)
            >>> model.train_faster_with_batch(batch_size=24, max_epochs=5, use_gpu=0)
            >>> model.save_model('pyrovel_test_saved_model', overwrite=True)
            >>> model = PyroVelocity.load_model('pyrovel_test_saved_model', adata, use_gpu=0)
            >>> posterior_samples = model.posterior_samples(model.adata, num_samples=30)
            >>> pretty_print_dict(posterior_samples)
            >>> print(posterior_samples.keys())
            >>> model = PyroVelocity(adata)
            >>> model.train(max_epochs=5, use_gpu=0)
            >>> posterior_samples = model.posterior_samples(model.adata, num_samples=30)
            >>> print(posterior_samples.keys())
            >>> model.save_model('pyrovel_test_saved_model', overwrite=True)
            >>> model = PyroVelocity.load_model('pyrovel_test_saved_model', adata, use_gpu=0)

        """
        # >>> assert isinstance(posterior_samples, dict), f"Expected a dictionary, got {type(posterior_samples)}"
        # >>> pretty_print_dict(posterior_samples)
        self.use_gpu = use_gpu
        self.cell_specific_kinetics = cell_specific_kinetics
        self.k = kinetics_num
        if input_type == "knn":
            layers = ["Mu", "Ms"]
            assert likelihood in {"Normal", "LogNormal"}
            assert "Mu" in adata.layers
        elif input_type == "raw_cpm":
            layers = ["unspliced", "spliced"]
            assert likelihood in {"Normal", "LogNormal"}
        else:
            layers = ["raw_unspliced", "raw_spliced"]
            assert likelihood != "Normal"

        self.layers = layers
        self.input_type = input_type

        # adata = setup_anndata_multilayers(
        #    adata,
        #    layer=self.layers,
        #    copy=True,
        #    batch_key=None,
        #    input_type=input_type,
        #    cluster=self.cell_specific_kinetics,
        # )
        # scvi.data.view_anndata_setup(adata)

        super().__init__(adata)
        if init:
            initial_values = init_with_all_cells(
                self.adata,
                input_type,
                shared_time,
                latent_factor,
                latent_factor_size,
                plate_size,
                num_aux_cells=num_aux_cells,
            )
        else:
            initial_values = {}
        logger.info(self.summary_stats)
        self.module = VelocityModule(
            self.summary_stats["n_cells"],
            self.summary_stats["n_vars"],
            model_type=model_type,
            guide_type=guide_type,
            likelihood=likelihood,
            shared_time=shared_time,
            t_scale_on=t_scale_on,
            plate_size=plate_size,
            latent_factor=latent_factor,
            latent_factor_operation=latent_factor_operation,
            latent_factor_size=latent_factor_size,
            inducing_point_size=inducing_point_size,
            include_prior=include_prior,
            use_gpu=use_gpu,
            num_aux_cells=num_aux_cells,
            only_cell_times=only_cell_times,
            decoder_on=decoder_on,
            add_offset=add_offset,
            correct_library_size=correct_library_size,
            cell_specific_kinetics=cell_specific_kinetics,
            kinetics_num=self.k,
            **initial_values,
        )
        self.num_cells = self.module.num_cells
        self._model_summary_string = """
        RNA velocity Pyro model with parameters:
        """
        self.init_params_ = self._get_init_params(locals())
        logger.info("The model has been initialized")

    def train(self, **kwargs):
        """
        Trains the PyroVelocity model using the provided data and configuration.

        The method leverages the Pyro library to train the model using the underlying
        data. It relies on the `VelocityTrainingMixin` to define the training logic.

        Args:

            **kwargs : dict, optional
                Additional keyword arguments to be passed to the underlying train method
                provided by the `VelocityTrainingMixin`.
        """
        pyro.enable_validation(True)
        super().train(**kwargs)

    # def predict_new_samples(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     batch_size: Optional[int] = None,
    #     num_samples: Optional[int] = 100,
    # ) -> Dict[str, ndarray]:
    #     adata = setup_anndata_multilayers(
    #         adata,
    #         layer=self.layers,
    #         copy=True,
    #         batch_key=None,
    #         input_type=self.input_type,
    #     )
    #     return self.posterior_samples(adata, indices, batch_size, num_samples)

    def enum_parallel_predict(self):
        """work for parallel enumeration"""
        return

    @classmethod
    def setup_anndata(cls, adata: AnnData, *args, **kwargs):
        """Latest scvi-tools interface"""
        setup_method_args = cls._get_setup_method_args(**locals())

        adata.obs["u_lib_size"] = np.log(adata.obs["u_lib_size_raw"] + 1e-6)
        adata.obs["s_lib_size"] = np.log(adata.obs["s_lib_size_raw"] + 1e-6)

        adata.obs["u_lib_size_mean"] = adata.obs["u_lib_size"].mean()
        adata.obs["s_lib_size_mean"] = adata.obs["s_lib_size"].mean()
        adata.obs["u_lib_size_scale"] = adata.obs["u_lib_size"].std()
        adata.obs["s_lib_size_scale"] = adata.obs["s_lib_size"].std()
        adata.obs["ind_x"] = np.arange(adata.n_obs).astype("int64")

        anndata_fields = [
            LayerField("U", "raw_unspliced", is_count_data=True),
            LayerField("X", "raw_spliced", is_count_data=True),
            NumericalObsField("u_lib_size", "u_lib_size"),
            NumericalObsField("s_lib_size", "s_lib_size"),
            NumericalObsField("u_lib_size_mean", "u_lib_size_mean"),
            NumericalObsField("s_lib_size_mean", "s_lib_size_mean"),
            NumericalObsField("u_lib_size_scale", "u_lib_size_scale"),
            NumericalObsField("s_lib_size_scale", "s_lib_size_scale"),
            NumericalObsField("ind_x", "ind_x"),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def posterior_samples(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        num_samples: int = 100,
    ) -> Dict[str, ndarray]:
        """
        If the guide uses sequential enumeration, computes the posterior samples for
        the given data using the trained PyroVelocity model.

        The method generates posterior samples by running the trained model on the
        provided data and returns a dictionary containing samples for each parameter.

        Args:
            adata (AnnData, optional): Anndata object containing the data for which posterior samples
                are to be computed. If not provided, the anndata used to initialize the model will be used.
            indices (Sequence[int], optional): Indices of cells in `adata` for which the posterior
                samples are to be computed.
            batch_size (int, optional): The size of the mini-batches used during computation.
                If not provided, the entire dataset will be used.
            num_samples (int, default: 100): The number of posterior samples to compute for each parameter.

        Returns:
            Dict[str, ndarray]: A dictionary containing the posterior samples for each parameter.
        """
        self.module.eval()
        predictive = self.module.create_predictive(
            model=pyro.poutine.uncondition(
                self.module.model
            ),  # do not input u_obs, and s_obs
            num_samples=num_samples,
        )

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        with torch.no_grad(), pyro.poutine.mask(mask=False):
            posterior_samples = []
            for tensor in scdl:
                args, kwargs = self.module._get_fn_args_from_batch(tensor)
                posterior_sample = {
                    k: v.cpu().numpy() for k, v in predictive(*args, **kwargs).items()
                }
                posterior_samples.append(posterior_sample)
            samples = {}
            for k in posterior_samples[0].keys():
                if k in [
                    "ut_norm",
                    "st_norm",
                    "cell_gene_state",
                ]:  # skip unused variables
                    continue

                if "aux" in k:
                    samples[k] = posterior_samples[0][k]  # alpha, beta, gamma...
                elif posterior_samples[0][k].shape[-2] == 1:
                    samples[k] = posterior_samples[0][k]
                    if k == "kinetics_prob":
                        samples[k] = np.concatenate(
                            [
                                # cat mini-batches
                                posterior_samples[j][k]
                                for j in range(len(posterior_samples))
                            ],
                            axis=-3,
                        )
                else:
                    samples[k] = np.concatenate(
                        [
                            # cat mini-batches
                            posterior_samples[j][k]
                            for j in range(len(posterior_samples))
                        ],
                        axis=-2,
                    )

                print(k, "before", sys.getsizeof(samples[k]))
                # if k in ['ut', 'st', 'u', 's']: # skip unused variables
                #    print(k, 'before', sys.getsizeof(samples[k]))
                #    samples[k] = sparse.csr_matrix(samples[k]) # cannot compress 3D tensor
                #    print(k, 'after', sys.getsizeof(samples[k]))
        self.num_samples = num_samples
        return samples

    def get_mlflow_logs(self):
        return

    def reduce_posterior_samples_dict(
        self,
        adata,
        model_posterior_samples,
        vector_field_basis,
        ncpus_use,
    ):
        """reduce posterior samples by precomputing metrics."""
        # adata = self._adata_manager.adata

        vector_field_posterior_samples, embeds_radian, fdri = vector_field_uncertainty(
            adata,
            model_posterior_samples,
            basis=vector_field_basis,
            n_jobs=ncpus_use,
        )
        embeds_magnitude = np.sqrt(
            (vector_field_posterior_samples**2).sum(axis=-1).sum(axis=-1)
        )

        mlflow.log_metric("FDR_sig_frac", round((fdri < 0.05).sum() / fdri.shape[0], 3))
        mlflow.log_metric("FDR_HMP", harmonic_mean(fdri))

        compute_mean_vector_field(
            pos=model_posterior_samples,
            adata=adata,
            basis=vector_field_basis,
            n_jobs=ncpus_use,
        )

        vector_field_posterior_mean = adata.obsm[f"velocity_pyro_{vector_field_basis}"]

        gene_ranking, genes = compute_volcano_data(
            [model_posterior_samples], [adata], time_correlation_with="st"
        )
        gene_ranking = (
            gene_ranking.sort_values("mean_mae", ascending=False)
            .head(300)
            .sort_values("time_correlation", ascending=False)
        )
        model_posterior_samples["gene_ranking"] = gene_ranking
        model_posterior_samples["genes"] = genes
        model_posterior_samples[
            "vector_field_posterior_samples"
        ] = vector_field_posterior_samples
        model_posterior_samples[
            "vector_field_posterior_mean"
        ] = vector_field_posterior_mean
        model_posterior_samples["fdri"] = fdri
        model_posterior_samples["embeds_magnitude"] = embeds_magnitude
        # assert embeds_radian.shape == (self.num_samples, adata.shape[0], 2)
        print(embeds_radian.shape)
        model_posterior_samples["embeds_angle"] = embeds_radian
        model_posterior_samples["ut_mean"] = (
            model_posterior_samples["ut"].mean(0).squeeze()
        )
        model_posterior_samples["st_mean"] = (
            model_posterior_samples["st"].mean(0).squeeze()
        )

        del model_posterior_samples["u"]
        del model_posterior_samples["s"]
        del model_posterior_samples["ut"]
        del model_posterior_samples["st"]
        return model_posterior_samples

    def save_prediction_pkl(self, model_posterior_samples, pyrovelocity_data_path):
        with open(pyrovelocity_data_path, "wb") as f:
            pickle.dump(model_posterior_samples, f)

    def save_model(
        self,
        dir_path: str,
        prefix: Optional[str] = None,
        overwrite: bool = True,
        save_anndata: bool = False,
        **anndata_write_kwargs,
    ) -> None:
        super().save(dir_path, prefix, overwrite, save_anndata, **anndata_write_kwargs)
        pyro.get_param_store().save(os.path.join(dir_path, "param_store_test.pt"))

    @classmethod
    def load_model(
        cls,
        dir_path: str,
        adata: Optional[AnnData] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        prefix: Optional[str] = None,
        backup_url: Optional[str] = None,
    ):
        load_adata = adata is None
        _, _, device = parse_use_gpu_arg(use_gpu)

        # scvi_setup_dict,
        (
            attr_dict,
            var_names,
            model_state_dict,
            new_adata,
        ) = _load_saved_files(
            dir_path,
            load_adata,
            map_location=device,
            prefix=prefix,
            backup_url=backup_url,
        )

        adata = new_adata if new_adata is not None else adata

        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        method_name = registry.get(_SETUP_METHOD_NAME, "setup_anndata")
        getattr(cls, method_name)(
            adata, source_registry=registry, **registry[_SETUP_ARGS_KEY]
        )

        model = _initialize_model(cls, adata, attr_dict)
        print("---------initiliaze-------")

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)
        print("setattr")

        # some Pyro modules with AutoGuides may need one training step
        pyro.clear_param_store()
        old_history = model.history_
        try:
            model.module.load_state_dict(model_state_dict)
        except RuntimeError as err:
            if not isinstance(model.module, PyroBaseModuleClass):
                raise err
            logger.info("Preparing underlying module for load")
            try:
                print("train 1---")
                model.train(max_epochs=1, max_steps=1, use_gpu=use_gpu)
            except Exception:
                model.train(
                    max_epochs=1,
                    max_steps=1,
                    use_gpu=use_gpu,
                    batch_size=adata.shape[0],
                )
            model.module.load_state_dict(model_state_dict)

        model.history_ = old_history
        print("load finished.")
        model.to_device(device)
        # avoid dropout prediction problem
        model.module.eval()
        model._validate_anndata(adata)
        # load pyro pyaram stores
        pyro.get_param_store().load(
            os.path.join(dir_path, "param_store_test.pt"), map_location=device
        )
        return model


# def _load_saved_files(
#    dir_path: str,
#    load_adata: bool,
#    map_location: Optional[Literal["cpu", "cuda"]] = None,
# ):
#    """Helper to load saved files, adapt from scvi-tools removing check in _CONSTANTS keys"""
#    setup_dict_path = os.path.join(dir_path, "attr.pkl")
#    adata_path = os.path.join(dir_path, "adata.h5ad")
#    varnames_path = os.path.join(dir_path, "var_names.csv")
#    model_path = os.path.join(dir_path, "model_params.pt")
#
#    if os.path.exists(adata_path) and load_adata:
#        adata = read(adata_path)
#    elif not os.path.exists(adata_path) and load_adata:
#        raise ValueError("Save path contains no saved anndata and no adata was passed.")
#    else:
#        adata = None
#
#    var_names = np.genfromtxt(varnames_path, delimiter=",", dtype=str)
#
#    with open(setup_dict_path, "rb") as handle:
#        attr_dict = pickle.load(handle)
#    scvi_setup_dict = attr_dict.pop("scvi_setup_dict_")
#    print(map_location)
#    model_state_dict = torch.load(model_path, map_location=map_location)
#    return scvi_setup_dict, attr_dict, var_names, model_state_dict, adata
