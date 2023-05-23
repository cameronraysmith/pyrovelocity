import argparse
import fnmatch
from pathlib import Path
from typing import Sequence
from typing import Union

import hydra
import rich.console
import rich.pretty
import rich.syntax
import rich.tree
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra_zen import make_config
from hydra_zen import make_custom_builds_fn
from omegaconf import DictConfig
from omegaconf import ListConfig
from omegaconf import OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from pyrovelocity.api import train_model
from pyrovelocity.utils import get_pylogger


def hydra_zen_configure():
    """
    Use hydra-zen to generate a configuration for hydra's config store.
    Generates a hydra-zen configuration for the pyrovelocity pipeline,
    which includes dataset, model training, and reporting configurations.

    This function defines helper functions for creating dataset, model,
    and reports configurations, and uses these functions to generate the
    hydra-zen configuration.

    Returns:
        Config : Type[DataClass]
        See the documentation for hydra_zen.make_config for more information.
    """
    # define helper functions
    pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

    def create_dataset_config(
        name, dl_root, data_file, rel_path, url, process_method, process_args
    ):
        return dict(
            data_file=data_file,
            dl_root=dl_root,
            dl_path="${.dl_root}/${.data_file}",
            rel_path=rel_path,
            url=url,
            derived=dict(
                process_method=process_method,
                process_args=process_args,
                rel_path="${data_external.processed_path}/" + f"{name}_processed.h5ad",
            ),
        )

    def create_model_config(
        source, name, model_suffix, vector_field_basis, **custom_training_parameters
    ):
        return dict(
            path="${paths.models}/" + f"{name}_model{model_suffix}",
            model_path="${.path}/model",
            input_data_path="${data_external."
            + f"{source}.{name}"
            + ".derived.rel_path}",
            trained_data_path="${.path}/trained.h5ad",
            pyrovelocity_data_path="${.path}/pyrovelocity.pkl",
            metrics_path="${.path}/metrics.json",
            run_info_path="${.path}/run_info.json",
            vector_field_parameters=dict(basis=vector_field_basis),
            training_parameters=pbuilds(
                train_model,
                loss_plot_path="${..path}/loss_plot.png",
                patient_improve=0.0001,
                patient_init=45,
                **custom_training_parameters,
            ),
        )

    def create_reports_config(model_name: str, model_number: int):
        path = f"{paths['reports']}/{model_name}_model{model_number}"
        return dict(
            path=path,
            dataframe_path=f"{paths['data']}/processed/{model_name}_model{model_number}_dataframe.pkl.zst",
            shared_time_plot=f"{path}/shared_time.pdf",
            volcano_plot=f"{path}/volcano.pdf",
            rainbow_plot=f"{path}/rainbow.pdf",
            uncertainty_param_plot=f"{path}/param_uncertainties.pdf",
            vector_field_plot=f"{path}/vector_field.pdf",
            biomarker_selection_plot=f"{path}/markers_selection_scatterplot.tif",
            biomarker_phaseportrait_plot=f"{path}/markers_phaseportrait.pdf",
        )

    # define configuration
    base = dict(log_level="INFO")

    paths = dict(
        data="data",
        models="models",
        reports="reports",
    )

    config = make_config(
        base=base,
        paths=paths,
        data_external=dict(
            root_path="${paths.data}/external",
            processed_path="${paths.data}/processed",
            sources=["simulate", "velocyto", "scvelo", "pyrovelocity"],
            simulate=dict(
                download=["medium"],
                process=["medium"],
                sources=dict(
                    gcs_root_url="https://storage.googleapis.com/pyrovelocity/data"
                ),
                medium=create_dataset_config(
                    "simulated_medium",
                    dl_root="${data_external.root_path}",
                    data_file="simulated_medium.h5ad",
                    rel_path="${data_external.root_path}/simulated_medium.h5ad",
                    url="${data_external.simulate.sources.gcs_root_url}/simulated_medium.h5ad",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            velocyto=dict(
                download=["pons"],
                process=["pons"],
                sources=dict(
                    gcs_root_url="https://storage.googleapis.com/pyrovelocity/data"
                ),
                pons=create_dataset_config(
                    "pons",
                    dl_root="${data_external.root_path}",
                    data_file="oligo_lite.h5ad",
                    rel_path="${data_external.root_path}/oligo_lite.h5ad",
                    url="${data_external.velocyto.sources.gcs_root_url}/oligo_lite.h5ad",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            scvelo=dict(
                download=["pancreas", "pbmc68k"],
                process=["pancreas", "pbmc68k"],
                sources=dict(
                    figshare_root_url="https://ndownloader.figshare.com/files",
                    scvelo_root_url="https://github.com/theislab/scvelo_notebooks/raw/master",
                ),
                pancreas=create_dataset_config(
                    "pancreas",
                    dl_root="data/Pancreas",
                    data_file="endocrinogenesis_day15.h5ad",
                    rel_path="${data_external.root_path}/endocrinogenesis_day15.h5ad",
                    url="${data_external.scvelo.sources.scvelo_root_url}/data/Pancreas/endocrinogenesis_day15.h5ad",
                    process_method="load_data",
                    process_args=dict(process_cytotrace=True),
                ),
                pbmc68k=create_dataset_config(
                    "pbmc68k",
                    dl_root="data/PBMC",
                    data_file="pbmc68k.h5ad",
                    rel_path="${data_external.root_path}/pbmc68k.h5ad",
                    url="${data_external.scvelo.sources.figshare_root_url}/27686886",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
            pyrovelocity=dict(
                download=["larry", "pbmc10k"],
                process=["pbmc10k"],
                sources=dict(
                    figshare_root_url="https://ndownloader.figshare.com/files"
                ),
                larry=create_dataset_config(
                    "larry",
                    dl_root="${data_external.root_path}",
                    data_file="larry.h5ad",
                    rel_path="${data_external.root_path}/larry.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028569",
                    process_method="load_data",
                    process_args=dict(),
                ),
                larry_tips=create_dataset_config(
                    "larry_tips",
                    dl_root="${data_external.root_path}",
                    data_file="larry_tips.h5ad",
                    rel_path="${data_external.root_path}/larry_tips.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028569",
                    process_method="load_data",
                    process_args=dict(),
                ),
                larry_mono=create_dataset_config(
                    "larry_mono",
                    dl_root="${data_external.root_path}",
                    data_file="larry_mono.h5ad",
                    rel_path="${data_external.root_path}/larry_mono.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028572",
                    process_method="load_data",
                    process_args=dict(),
                ),
                larry_neu=create_dataset_config(
                    "larry_neu",
                    dl_root="${data_external.root_path}",
                    data_file="larry_neu.h5ad",
                    rel_path="${data_external.root_path}/larry_neu.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028575",
                    process_method="load_data",
                    process_args=dict(),
                ),
                larry_multilineage=create_dataset_config(
                    "larry_multilineage",
                    dl_root="${data_external.root_path}",
                    data_file="larry_mono.h5ad",
                    rel_path="${data_external.root_path}/larry_mono.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/37028572",
                    process_method="load_data",
                    process_args=dict(),
                ),
                pbmc10k=create_dataset_config(
                    "pbmc10k",
                    dl_root="${data_external.root_path}",
                    data_file="pbmc10k.h5ad",
                    rel_path="${data_external.root_path}/pbmc10k.h5ad",
                    url="${data_external.pyrovelocity.sources.figshare_root_url}/pbmc10k",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
        ),
        model_training=dict(
            train=[
                "simulate_model1",
                "simulate_model2",
                "pancreas_model1",
                "pancreas_model2",
                "pbmc68k_model1",
                "pbmc68k_model2",
                "pons_model1",
                "pons_model2",
                "larry_model2",
                "larry_tips_model2",
                "larry_mono_model2",
                "larry_neu_model2",
                "larry_multilineage_model2",
                "pbmc10k_model2",
            ],
            simulate_model1=create_model_config(
                "simulate",
                "medium",
                1,
                "umap",
                cell_state="leiden",
                guide_type="auto_t0_constraint",
                max_epochs=4000,
            ),
            simulate_model2=create_model_config(
                "simulate",
                "medium",
                2,
                "umap",
                cell_state="leiden",
                max_epochs=4000,
                offset=True,
            ),
            pancreas_model1=create_model_config(
                "scvelo",
                "pancreas",
                1,
                "umap",
                guide_type="auto_t0_constraint",
                max_epochs=2000,
            ),
            pancreas_model2=create_model_config(
                "scvelo",
                "pancreas",
                2,
                "umap",
                offset=True,
                max_epochs=2000,
            ),
            pbmc68k_model1=create_model_config(
                "scvelo",
                "pbmc68k",
                1,
                "tsne",
                guide_type="auto_t0_constraint",
                cell_state="celltype",
                max_epochs=2000,
            ),
            pbmc68k_model2=create_model_config(
                "scvelo",
                "pbmc68k",
                2,
                "tsne",
                cell_state="celltype",
                offset=True,
                max_epochs=2000,
            ),
            pons_model1=create_model_config(
                "velocyto",
                "pons",
                1,
                "umap",
                guide_type="auto_t0_constraint",
                cell_state="celltype",
                max_epochs=2000,
            ),
            pons_model2=create_model_config(
                "velocyto",
                "pons",
                2,
                "umap",
                cell_state="celltype",
                offset=True,
                max_epochs=2000,
            ),
            larry_model2=create_model_config(
                "pyrovelocity",
                "larry",
                2,
                "emb",
                svi_train=True,
                batch_size=4000,
                cell_state="state_info",
                offset=True,
                max_epochs=1000,
            ),
            larry_tips_model2=create_model_config(
                "pyrovelocity",
                "larry_tips",
                2,
                "umap",
                svi_train=True,
                batch_size=4000,
                cell_state="state_info",
                offset=True,
                max_epochs=1000,
            ),
            larry_mono_model2=create_model_config(
                "pyrovelocity",
                "larry_mono",
                2,
                "emb",
                svi_train=True,
                batch_size=4000,
                cell_state="state_info",
                offset=True,
                max_epochs=1000,
            ),
            larry_neu_model2=create_model_config(
                "pyrovelocity",
                "larry_neu",
                2,
                "emb",
                svi_train=True,
                batch_size=4000,
                cell_state="state_info",
                offset=True,
                max_epochs=1000,
            ),
            larry_multilineage_model2=create_model_config(
                "pyrovelocity",
                "larry_multilineage",
                2,
                "emb",
                svi_train=True,
                batch_size=4000,
                cell_state="state_info",
                offset=True,
                max_epochs=1000,
            ),
            pbmc10k_model2=create_model_config(
                "pyrovelocity",
                "pbmc10k",
                2,
                "tsne",
                cell_state="leiden",
                offset=True,
                max_epochs=2000,
            ),
        ),
        reports=dict(
            model_summary=dict(
                summarize=[
                    "simulate_model1",
                    "simulate_model2",
                    "pancreas_model1",
                    "pancreas_model2",
                    "pbmc68k_model1",
                    "pbmc68k_model2",
                    "pons_model1",
                    "pons_model2",
                    "pbmc10k_model2",
                    "larry_tips_model2",
                ],
                simulate_model1=create_reports_config("medium", 1),
                simulate_model2=create_reports_config("medium", 2),
                pancreas_model1=create_reports_config("pancreas", 1),
                pancreas_model2=create_reports_config("pancreas", 2),
                pbmc68k_model1=create_reports_config("pbmc68k", 1),
                pbmc68k_model2=create_reports_config("pbmc68k", 2),
                pons_model1=create_reports_config("pons", 1),
                pons_model2=create_reports_config("pons", 2),
                pbmc10k_model2=create_reports_config("pbmc10k", 2),
                larry_tips_model2=create_reports_config("larry_tips", 2),
            ),
            figure2=dict(
                tag="fig2",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model1.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model1.svg",
                biomarker_selection_path="${.path}/${.tag}_markers_selection_scatterplot.tif",
                biomarker_phaseportrait_path="${.path}/${.tag}_markers_phaseportrait.pdf",
                uncertainty_param_plot_path="${.path}/${.tag}_param_uncertainties.pdf",
                uncertainty_magnitude_plot_path="${.path}/${.tag}_magnitude_uncertainties.pdf",
                pancreas_model1=dict(
                    shared_time_plot="${..path}/${..tag}_pancreas_shared_time.pdf",
                    volcano_plot="${..path}/${..tag}_pancreas_volcano.pdf",
                    rainbow_plot="${..path}/${..tag}_pancreas_rainbow.pdf",
                    vector_field_plot="${..path}/${..tag}_pancreas_vector_field.pdf",
                ),
            ),
            figure2_extras=dict(
                tag="fig2",
                path="${paths.reports}/${.tag}",
                shared_time_plot="${.path}/figure2_extras.pdf",
            ),
            figureS3=dict(
                tag="figS3",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model2.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model2.svg",
            ),
            figureS3_extras=dict(
                tag="figS3",
                path="${paths.reports}/${.tag}",
                violin_plots_pbmc_lin="${.path}/figureS3_extras_pbmc_lin.pdf",
                violin_plots_pbmc_log="${.path}/figureS3_extras_pbmc_log.pdf",
                violin_plots_larry_lin="${.path}/figureS3_extras_larry_lin.pdf",
                violin_plots_larry_log="${.path}/figureS3_extras_larry_log.pdf",
            ),
        ),
    )
    return config


def hydra_zen_compressed_configure():
    """
    Use hydra-zen to generate a configuration for hydra's config store.
    Generates a hydra-zen configuration for the pyrovelocity pipeline,
    which includes dataset, model training, and reporting configurations.

    This function defines helper functions for creating dataset, model,
    and reports configurations, and uses these functions to generate the
    hydra-zen configuration.

    Returns:
        Config : Type[DataClass]
        See the documentation for hydra_zen.make_config for more information.
    """
    # define helper functions
    pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

    def create_dataset_config(
        source, name, dl_root, data_file, rel_path, url, process_method, process_args
    ):
        return dict(
            source=source,
            data_file=data_file,
            dl_root=dl_root,
            dl_path="${.dl_root}/${.data_file}",
            rel_path=rel_path,
            url=url,
            derived=dict(
                process_method=process_method,
                process_args=process_args,
                rel_path="${paths.data_processed}/" + f"{name}_processed.h5ad",
                thresh_histogram_path="${paths.data_processed}/"
                + f"{name}_thresh_histogram.pdf",
            ),
        )

    def create_model_config(
        source,
        name,
        model_suffix,
        vector_field_basis,
        gpu_id=0,
        **custom_training_parameters,
    ):
        return dict(
            path="${paths.models}/" + f"{name}_model{model_suffix}",
            model_path="${.path}/model",
            input_data_path="${data_sets." + f"{name}" + ".derived.rel_path}",
            trained_data_path="${.path}/trained.h5ad",
            pyrovelocity_data_path="${.path}/pyrovelocity.pkl.zst",
            posterior_samples_path="${.path}/posterior_samples.pkl.zst",
            metrics_path="${.path}/metrics.json",
            run_info_path="${.path}/run_info.json",
            vector_field_parameters=dict(basis=vector_field_basis),
            gpu_id=gpu_id,
            training_parameters=pbuilds(
                train_model,
                loss_plot_path="${..path}/loss_plot.png",
                patient_improve=0.0001,
                patient_init=45,
                **custom_training_parameters,
            ),
        )

    def create_reports_config(model_name: str, model_number: int):
        path = f"{paths['reports']}/{model_name}_model{model_number}"
        return dict(
            path=path,
            trained_data_path="${paths.models}/"
            + f"{model_name}_model{model_number}/trained.h5ad",
            pyrovelocity_data_path="${paths.models}/"
            + f"{model_name}_model{model_number}/pyrovelocity.pkl.zst",
            dataframe_path=f"{paths['data']}/processed/{model_name}_model{model_number}_dataframe.pkl.zst",
            shared_time_plot=f"{path}/shared_time.pdf",
            volcano_plot=f"{path}/volcano.pdf",
            rainbow_plot=f"{path}/rainbow.pdf",
            uncertainty_param_plot=f"{path}/param_uncertainties.pdf",
            vector_field_plot=f"{path}/vector_field.pdf",
            biomarker_selection_plot=f"{path}/markers_selection_scatterplot.tif",
            biomarker_phaseportrait_plot=f"{path}/markers_phaseportrait.pdf",
            fig2_part1_plot=f"{path}/fig2_part1_plot.pdf",
            fig2_part2_plot=f"{path}/fig2_part2_plot.pdf",
        )

    base = dict(log_level="INFO")

    paths = dict(
        data="data",
        models="models",
        reports="reports",
        data_external="${paths.data}/external",
        data_processed="${paths.data}/processed",
    )

    download_data = [
        "simulate_medium",
        "pons",
        "pancreas",
        "pbmc10k",
        "pbmc68k",
        "larry",
        "larry_cospar",
        "larry_cytotrace",
        "larry_dynamical",
    ]

    process_data = [
        "simulate_medium",
        "pons",
        "pancreas",
        "pbmc10k",
        "pbmc68k",
        "larry",
        "larry_tips",
        "larry_mono",
        "larry_neu",
        "larry_multilineage",
    ]

    train_models = [
        "simulate_medium_model1",
        "simulate_medium_model2",
        "pancreas_model1",
        "pancreas_model2",
        "pbmc68k_model1",
        "pbmc68k_model2",
        "pons_model1",
        "pons_model2",
        "larry_model2",
        "larry_tips_model2",
        "larry_mono_model2",
        "larry_neu_model2",
        "larry_multilineage_model2",
        "pbmc10k_model2",
    ]

    model_training = dict(
        simulate_model1=create_model_config(
            "simulate",
            "simulate_medium",
            1,
            "umap",
            cell_state="leiden",
            guide_type="auto_t0_constraint",
            max_epochs=4000,
        ),
        simulate_model2=create_model_config(
            "simulate",
            "simulate_medium",
            2,
            "umap",
            cell_state="leiden",
            max_epochs=300,
            offset=True,
        ),
        pancreas_model1=create_model_config(
            "scvelo",
            "pancreas",
            1,
            "umap",
            guide_type="auto_t0_constraint",
            max_epochs=2000,
        ),
        pancreas_model2=create_model_config(
            "scvelo",
            "pancreas",
            2,
            "umap",
            gpu_id=0,
            offset=True,
            max_epochs=2000,
        ),
        pbmc68k_model1=create_model_config(
            "scvelo",
            "pbmc68k",
            1,
            "tsne",
            guide_type="auto_t0_constraint",
            cell_state="celltype",
            max_epochs=2000,
        ),
        pbmc68k_model2=create_model_config(
            "scvelo",
            "pbmc68k",
            2,
            "tsne",
            gpu_id=1,
            cell_state="celltype",
            offset=True,
            max_epochs=2000,
        ),
        pons_model1=create_model_config(
            "velocyto",
            "pons",
            1,
            "umap",
            guide_type="auto_t0_constraint",
            cell_state="celltype",
            max_epochs=2000,
        ),
        pons_model2=create_model_config(
            "velocyto",
            "pons",
            2,
            "umap",
            gpu_id=2,
            cell_state="celltype",
            offset=True,
            max_epochs=2000,
        ),
        larry_model2=create_model_config(
            "pyrovelocity",
            "larry",
            2,
            "emb",
            gpu_id=3,
            svi_train=True,
            batch_size=4000,
            cell_state="state_info",
            offset=True,
            max_epochs=1000,
        ),
        larry_tips_model2=create_model_config(
            "pyrovelocity",
            "larry_tips",
            2,
            "umap",
            gpu_id=0,
            svi_train=True,
            batch_size=4000,
            cell_state="state_info",
            offset=True,
            max_epochs=1000,
        ),
        larry_mono_model2=create_model_config(
            "pyrovelocity",
            "larry_mono",
            2,
            "emb",
            gpu_id=1,
            svi_train=True,
            batch_size=4000,
            cell_state="state_info",
            offset=True,
            max_epochs=1000,
        ),
        larry_neu_model2=create_model_config(
            "pyrovelocity",
            "larry_neu",
            2,
            "emb",
            gpu_id=2,
            svi_train=True,
            batch_size=4000,
            cell_state="state_info",
            offset=True,
            max_epochs=1000,
        ),
        larry_multilineage_model2=create_model_config(
            "pyrovelocity",
            "larry_multilineage",
            2,
            "emb",
            gpu_id=3,
            svi_train=True,
            batch_size=4000,
            cell_state="state_info",
            offset=True,
            max_epochs=1000,
        ),
        pbmc10k_model2=create_model_config(
            "pyrovelocity",
            "pbmc10k",
            2,
            "tsne",
            gpu_id=0,
            cell_state="leiden",
            offset=True,
            max_epochs=2000,
        ),
    )

    data_sets = dict(
        simulate_medium=create_dataset_config(
            source="simulate",
            name="simulated_medium",
            dl_root="${paths.data_external}",
            data_file="simulated_medium.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/simulated_medium.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        pons=create_dataset_config(
            source="velocyto",
            name="pons",
            dl_root="${paths.data_external}",
            data_file="oligo_lite.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/oligo_lite.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        pancreas=create_dataset_config(
            source="scvelo",
            name="pancreas",
            dl_root="data/Pancreas",
            data_file="endocrinogenesis_day15.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad",
            process_method="load_data",
            process_args=dict(process_cytotrace=True, count_thres=4),
        ),
        pbmc68k=create_dataset_config(
            source="scvelo",
            name="pbmc68k",
            dl_root="data/PBMC",
            data_file="pbmc68k.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/27686886",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry=create_dataset_config(
            source="pyrovelocity",
            name="larry",
            dl_root="${paths.data_external}",
            data_file="larry.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/37028569",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_cospar=create_dataset_config(
            source="pyrovelocity",
            name="larry_cospar",
            dl_root="${paths.data_external}",
            data_file="larry_cospar.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/larry_cospar.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_cytotrace=create_dataset_config(
            source="pyrovelocity",
            name="larry_cytotrace",
            dl_root="${paths.data_external}",
            data_file="larry_cytotrace.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/larry_cytotrace.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_dynamical=create_dataset_config(
            source="pyrovelocity",
            name="larry_dynamical",
            dl_root="${paths.data_external}",
            data_file="larry_dynamical.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/larry_dynamical.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_tips=create_dataset_config(
            source="pyrovelocity",
            name="larry_tips",
            dl_root="${paths.data_external}",
            data_file="larry_tips.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/37028569",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_mono=create_dataset_config(
            source="pyrovelocity",
            name="larry_mono",
            dl_root="${paths.data_external}",
            data_file="larry_mono.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/37028569",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_neu=create_dataset_config(
            source="pyrovelocity",
            name="larry_neu",
            dl_root="${paths.data_external}",
            data_file="larry_neu.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/37028575",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        larry_multilineage=create_dataset_config(
            source="pyrovelocity",
            name="larry_multilineage",
            dl_root="${paths.data_external}",
            data_file="larry_multilineage.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://ndownloader.figshare.com/files/37028569",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
        pbmc10k=create_dataset_config(
            source="pyrovelocity",
            name="pbmc10k",
            dl_root="${paths.data_external}",
            data_file="pbmc10k.h5ad",
            rel_path="${paths.data_external}/${.data_file}",
            url="https://storage.googleapis.com/pyrovelocity/data/pbmc10k.h5ad",
            process_method="load_data",
            process_args=dict(count_thres=4),
        ),
    )

    return make_config(
        base=base,
        paths=paths,
        download_data=download_data,
        process_data=process_data,
        train_models=train_models,
        data_sets=data_sets,
        model_training={
            k: model_training[k]
            for k in model_training
            if fnmatch.fnmatch(k, "*_model2")
        },
        reports=dict(
            model_summary=dict(
                # simulate_model1=create_reports_config("medium", 1),
                # simulate_model2=create_reports_config("medium", 2),
                # pancreas_model1=create_reports_config("pancreas", 1),
                pancreas_model2=create_reports_config("pancreas", 2),
                # pbmc68k_model1=create_reports_config("pbmc68k", 1),
                pbmc68k_model2=create_reports_config("pbmc68k", 2),
                # pons_model1=create_reports_config("pons", 1),
                pons_model2=create_reports_config("pons", 2),
                pbmc10k_model2=create_reports_config("pbmc10k", 2),
                larry_tips_model2=create_reports_config("larry_tips", 2),
                larry_model2=create_reports_config("larry", 2),
            ),
            figure2=dict(
                tag="fig2",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model1.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model1.svg",
                biomarker_selection_path="${.path}/${.tag}_markers_selection_scatterplot.tif",
                biomarker_phaseportrait_path="${.path}/${.tag}_markers_phaseportrait.pdf",
                uncertainty_param_plot_path="${.path}/${.tag}_param_uncertainties.pdf",
                uncertainty_magnitude_plot_path="${.path}/${.tag}_magnitude_uncertainties.pdf",
                pancreas_model1=dict(
                    shared_time_plot="${..path}/${..tag}_pancreas_shared_time.pdf",
                    volcano_plot="${..path}/${..tag}_pancreas_volcano.pdf",
                    rainbow_plot="${..path}/${..tag}_pancreas_rainbow.pdf",
                    vector_field_plot="${..path}/${..tag}_pancreas_vector_field.pdf",
                ),
            ),
            figure2_extras=dict(
                tag="fig2",
                path="${paths.reports}/${.tag}",
                shared_time_plot="${.path}/figure2_extras.pdf",
            ),
            figureS3=dict(
                tag="figS3",
                path="${paths.reports}/${.tag}",
                tif_path="${.path}/${.tag}_raw_gene_selection_model2.tif",
                svg_path="${.path}/${.tag}_raw_gene_selection_model2.svg",
            ),
            figureS3_extras=dict(
                tag="figS3",
                path="${paths.reports}/${.tag}",
                violin_plots_pbmc_lin="${.path}/figureS3_extras_pbmc_lin.pdf",
                violin_plots_pbmc_log="${.path}/figureS3_extras_pbmc_log.pdf",
                violin_plots_larry_lin="${.path}/figureS3_extras_larry_lin.pdf",
                violin_plots_larry_log="${.path}/figureS3_extras_larry_log.pdf",
            ),
        ),
    )


def test_hydra_zen_configure():
    """
    Use hydra-zen to generate a test configuration for hydra's config store.
    Generates a hydra-zen configuration to test the pyrovelocity pipeline,
    which includes dataset, model training, and reporting configurations.

    This function defines helper functions for creating dataset, model,
    and reports configurations, and uses these functions to generate the
    hydra-zen configuration.

    Returns:
        Config : Type[DataClass]
        See the documentation for hydra_zen.make_config for more information.
    """
    pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

    def create_dataset_config(
        name, dl_root, data_file, rel_path, url, process_method, process_args
    ):
        return dict(
            data_file=data_file,
            dl_root=dl_root,
            dl_path="${.dl_root}/${.data_file}",
            rel_path=rel_path,
            url=url,
            derived=dict(
                process_method=process_method,
                process_args=process_args,
                rel_path="${data_external.processed_path}/" + f"{name}_processed.h5ad",
            ),
        )

    def create_model_config(
        source, name, model_suffix, vector_field_basis, **custom_training_parameters
    ):
        return dict(
            path="${paths.models}/" + f"{name}_model{model_suffix}",
            model_path="${.path}/model",
            input_data_path="${data_external."
            + f"{source}.{name}"
            + ".derived.rel_path}",
            trained_data_path="${.path}/trained.h5ad",
            pyrovelocity_data_path="${.path}/pyrovelocity.pkl",
            metrics_path="${.path}/metrics.json",
            run_info_path="${.path}/run_info.json",
            vector_field_parameters=dict(basis=vector_field_basis),
            training_parameters=pbuilds(
                train_model,
                loss_plot_path="${..path}/loss_plot.png",
                patient_improve=0.0001,
                patient_init=45,
                **custom_training_parameters,
            ),
        )

    def create_reports_config(model_name: str, model_number: int):
        path = f"{paths['reports']}/{model_name}_model{model_number}"
        return dict(
            path=path,
            dataframe_path=f"{paths['data']}/processed/{model_name}_model{model_number}_dataframe.pkl.zst",
            shared_time_plot=f"{path}/shared_time.pdf",
            volcano_plot=f"{path}/volcano.pdf",
            rainbow_plot=f"{path}/rainbow.pdf",
            uncertainty_param_plot=f"{path}/param_uncertainties.pdf",
            vector_field_plot=f"{path}/vector_field.pdf",
            biomarker_selection_plot=f"{path}/markers_selection_scatterplot.tif",
            biomarker_phaseportrait_plot=f"{path}/markers_phaseportrait.pdf",
        )

    base = dict(log_level="INFO")

    paths = dict(
        data="data",
        models="models",
        reports="reports",
    )

    config = make_config(
        base=base,
        paths=paths,
        data_external=dict(
            root_path="${paths.data}/external",
            processed_path="${paths.data}/processed",
            sources=["simulate"],
            simulate=dict(
                download=["medium"],
                process=["medium"],
                sources=dict(
                    gcs_root_url="https://storage.googleapis.com/pyrovelocity/data"
                ),
                medium=create_dataset_config(
                    "simulated_medium",
                    dl_root="${data_external.root_path}",
                    data_file="simulated_medium.h5ad",
                    rel_path="${data_external.root_path}/simulated_medium.h5ad",
                    url="${data_external.simulate.sources.gcs_root_url}/simulated_medium.h5ad",
                    process_method="load_data",
                    process_args=dict(),
                ),
            ),
        ),
        model_training=dict(
            train=[
                "simulate_model1",
                "simulate_model2",
            ],
            simulate_model1=create_model_config(
                "simulate",
                "medium",
                1,
                "umap",
                cell_state="leiden",
                guide_type="auto_t0_constraint",
                max_epochs=1000,
            ),
            simulate_model2=create_model_config(
                "simulate",
                "medium",
                2,
                "umap",
                cell_state="leiden",
                max_epochs=1000,
                offset=True,
            ),
        ),
        reports=dict(
            model_summary=dict(
                summarize=[
                    "simulate_model1",
                    "simulate_model2",
                ],
                simulate_model1=create_reports_config("medium", 1),
                simulate_model2=create_reports_config("medium", 2),
            ),
        ),
    )
    return config


def initialize_hydra_config() -> DictConfig:
    """Initialize Hydra configuration for PyroVelocity pipeline.

    Returns:
        DictConfig: A Hydra configuration object containing dataset, model training,
        and reporting configurations for the pyrovelocity pipeline.

    Example:
        >>> cfg = initialize_hydra_config()
    """
    GlobalHydra.instance().clear()
    config = hydra_zen_configure()
    hydra.initialize(version_base="1.2", config_path=None)
    cs = ConfigStore.instance()
    cs.store(name="config", node=config)
    return hydra.compose(config_name="config")


def config_setup(config_path: str) -> Union[DictConfig, ListConfig]:
    """Convert template into concrete configuration file.
    Args:
        config_path {Text}: path to config
    """

    logger = get_pylogger(name="CONF")

    template_config_path = config_path.replace("config.yaml", "template-config.yaml")
    conf = OmegaConf.load(template_config_path)
    with open(config_path, "w") as conf_file:
        OmegaConf.save(config=conf, f=conf_file, resolve=True)

    conf = OmegaConf.load(config_path)
    print_config_tree(conf, logger, ())
    return conf


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    logger,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    config_setup(config_path=args.config)
