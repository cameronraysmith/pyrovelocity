import os
import pickle
from logging import Logger
from pathlib import Path
from statistics import harmonic_mean
from typing import Text

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
import seaborn as sns
from omegaconf import DictConfig

# from astropy import units as u
# from astropy.stats import circstd
from scipy.stats import circmean
from scipy.stats import circstd
from scipy.stats import circvar

from pyrovelocity.config import print_config_tree
from pyrovelocity.data import load_data
from pyrovelocity.io.compressedpickle import CompressedPickle
from pyrovelocity.plot import compute_mean_vector_field
from pyrovelocity.plot import compute_volcano_data
from pyrovelocity.plot import plot_arrow_examples
from pyrovelocity.plot import plot_gene_ranking
from pyrovelocity.plot import plot_posterior_time
from pyrovelocity.plot import plot_vector_field_uncertain
from pyrovelocity.plot import rainbowplot
from pyrovelocity.plot import us_rainbowplot
from pyrovelocity.plot import vector_field_uncertainty
from pyrovelocity.utils import anndata_counts_to_df
from pyrovelocity.utils import get_pylogger
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.utils import print_anndata
from pyrovelocity.utils import print_attributes


"""Loads model-trained data and generates figures.

for data_model in
    [
        pancreas_model1,
        pancreas_model2,
        pbmc68k_model1,
        pbmc68k_model2,
        pons_model1,
        pons_model2,
    ]

Inputs:
  data:
    "models/{data_model}/trained.h5ad"
    "models/{data_model}/pyrovelocity.pkl"

Outputs:
  figures:
    shared_time_plot: reports/{data_model}/shared_time.pdf
    volcano_plot: reports/{data_model}/volcano.pdf
    rainbow_plot: reports/{data_model}/rainbow.pdf
    uncertainty_param_plot: reports/{data_model}/param_uncertainties.pdf
    vector_field_plot: reports/{data_model}/vector_field.pdf
    biomarker_selection_plot: reports/{data_model}/markers_selection_scatterplot.tif
    biomarker_phaseportrait_plot: reports/{data_model}/markers_phaseportrait.pdf
"""


def summarize_fig2_part1(
    adata,
    posterior_vector_field,
    posterior_time,
    cell_magnitudes,
    pca_embeds_angle,
    embed_radians,
    embedding,
    embed_mean,
    cluster="cell_type",
    plot_name="test",
):
    dot_size = 3.5
    font_size = 6.5
    scale = 0.35
    # scale_high = 0.05
    # scale_low = 0.009
    scale_high = 7.8
    scale_low = 7.8

    arrow = 3.6
    density = 0.4
    ress = pd.DataFrame(
        {
            "cell_type": adata.obs[cluster].values,
            "X1": adata.obsm[f"X_{embedding}"][:, 0],
            "X2": adata.obsm[f"X_{embedding}"][:, 1],
        }
    )
    fig = plt.figure(figsize=(9.6, 2), constrained_layout=False)
    fig.subplots_adjust(
        hspace=0.2, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.45
    )
    ax = fig.subplots(1, 6)
    pos = ax[0].get_position()

    sns.scatterplot(
        x="X1",
        y="X2",
        data=ress,
        alpha=0.9,
        s=dot_size,
        linewidth=0,
        edgecolor="none",
        hue="cell_type",
        ax=ax[0],
        legend="brief",
    )
    ax[0].axis("off")
    ax[0].set_title("Cell types\n", fontsize=font_size)
    print(pos.x0, pos.x1)
    ax[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.48),
        ncol=5,
        fancybox=True,
        prop={"size": font_size},
        fontsize=font_size,
        frameon=False,
    )
    kwargs = dict(
        color="gray",
        s=dot_size,
        show=False,
        alpha=0.25,
        min_mass=3.5,
        scale=scale,
        frameon=False,
        density=density,
        arrow_size=3,
        linewidth=1,
    )
    scv.pl.velocity_embedding_grid(
        adata, basis=embedding, fontsize=font_size, ax=ax[1], title="", **kwargs
    )
    ax[1].set_title("Scvelo\n", fontsize=7)
    scv.pl.velocity_embedding_grid(
        adata,
        fontsize=font_size,
        basis=embedding,
        title="",
        ax=ax[2],
        vkey="velocity_pyro",
        **kwargs,
    )
    ax[2].set_title("Pyro-Velocity\n", fontsize=7)

    # astropy degree angular deviation is buggy...
    # pca_cell_angles = pca_embeds_angle / np.pi * 180 # degree
    # pca_cell_angles_mean = pca_cell_angles.mean(axis=0)
    # pca_angles_std = circstd(pca_cell_angles * u.deg, method="angular", axis=0)
    # pca_cell_angles_cov = pca_angles_std / pca_cell_angles_mean
    # scipy version circstd [0, inf]
    pca_cell_radians = pca_embeds_angle
    # pca_cell_cirstd = circstd(pca_cell_radians, axis=0)
    # scipy circvar [0, 1]
    pca_cell_circov = circvar(pca_cell_radians, axis=0)

    #    plot_arrow_examples(
    #        adata,
    #        np.transpose(posterior_vector_field, (1, 2, 0)),
    #        embed_radians,
    #        embed_mean,
    #        ax=ax[3],
    #        n_sample=20,
    #        fig=fig,
    #        basis=embedding,
    #        scale=scale_high,
    #        alpha=0.2,
    #        index=3,
    #        index2=5,
    #        scale2=scale_low,
    #        num_certain=2, #TODO: add parameter for selecting arrows
    #        num_total=4,
    #        density=density,
    #        arrow_size=arrow+0.2,
    #        customize_uncertain=pca_cell_circov
    #    )
    #    ax[3].set_title("Single cell\nvector field", fontsize=7)

    cell_time_mean = posterior_time.mean(0).flatten()
    cell_time_std = posterior_time.std(0).flatten()
    cell_time_cov = cell_time_std / cell_time_mean

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_time_cov,
        ax=ax[3],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="shared time",
        cmap="winter",
        cmax=None,
    )

    cell_magnitudes_mean = cell_magnitudes.mean(axis=-2)
    cell_magnitudes_std = cell_magnitudes.std(axis=-2)
    cell_magnitudes_cov = cell_magnitudes_std / cell_magnitudes_mean
    plot_vector_field_uncertain(
        adata,
        embed_mean,
        cell_magnitudes_cov,
        ax=ax[4],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="base magnitude",
        cmap="summer",
        cmax=None,
    )

    plot_vector_field_uncertain(
        adata,
        embed_mean,
        pca_cell_circov,
        ax=ax[5],
        cbar=True,
        fig=fig,
        basis=embedding,
        scale=scale,
        arrow_size=arrow,
        p_mass_min=1,
        autoscale=True,
        density=density,
        only_grid=False,
        uncertain_measure="PCA angle",
        cmap="inferno",
        cmax=None,
    )
    for ext in ["", ".png"]:
        fig.savefig(
            f"{plot_name}{ext}",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )


def summarize_fig2_part2(
    adata, posterior_samples, plot_name="", basis="", cell_state="", fig=None
):
    if fig is None:
        fig = plt.figure(figsize=(9.5, 5))
        subfigs = fig.subfigures(1, 2, wspace=0.0, hspace=0, width_ratios=[1.8, 4])
        ax = subfigs[0].subplots(2, 1)
        plot_posterior_time(
            posterior_samples,
            adata,
            ax=ax[0],
            fig=subfigs[0],
            addition=False,
            basis=basis,
        )
        volcano_data, _ = plot_gene_ranking(
            [posterior_samples], [adata], ax=ax[1], time_correlation_with="st"
        )
        print(volcano_data.head())
        _ = rainbowplot(
            volcano_data,
            adata,
            posterior_samples,
            subfigs[1],
            data=["st", "ut"],
            basis=basis,
            cell_state=cell_state,
            num_genes=4,
        )
        for ext in ["", ".png"]:
            fig.savefig(
                f"{plot_name}{ext}",
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )


def plots(conf: DictConfig, logger: Logger) -> None:
    """Construct summary plots for each data set and model.

    Args:
        conf (DictConfig): OmegaConf configuration object
        logger (Logger): Python logger

    Examples:
        plots(conf, logger)
    """
    for data_model in conf.train_models:
        ##################
        # load data
        ##################

        data_model_conf = conf.model_training[data_model]
        cell_state = data_model_conf.training_parameters.cell_state
        trained_data_path = data_model_conf.trained_data_path
        pyrovelocity_data_path = data_model_conf.pyrovelocity_data_path
        reports_data_model_conf = conf.reports.model_summary[data_model]
        trained_data_path = reports_data_model_conf.trained_data_path
        pyrovelocity_data_path = reports_data_model_conf.pyrovelocity_data_path

        # print_config_tree(reports_data_model_conf, logger, ())

        logger.info(f"\n\nPlotting summary figure(s) in: {data_model}\n\n")

        logger.info(
            f"\n\nVerifying existence of path for:\n\n"
            f"  reports: {reports_data_model_conf.path}\n"
        )
        Path(reports_data_model_conf.path).mkdir(parents=True, exist_ok=True)

        dataframe_path = reports_data_model_conf.dataframe_path
        volcano_plot = reports_data_model_conf.volcano_plot
        rainbow_plot = reports_data_model_conf.rainbow_plot
        vector_field_plot = reports_data_model_conf.vector_field_plot
        shared_time_plot = reports_data_model_conf.shared_time_plot
        fig2_part1_plot = reports_data_model_conf.fig2_part1_plot
        fig2_part2_plot = reports_data_model_conf.fig2_part2_plot

        output_filenames = [
            dataframe_path,
            volcano_plot,
            rainbow_plot,
            vector_field_plot,
            shared_time_plot,
        ]
        if all(os.path.isfile(f) for f in output_filenames):
            logger.info(
                "\n\t" + "\n\t".join(output_filenames) + "\nAll output files exist"
            )
            return logger.warn("Remove output files if you want to regenerate them.")

        logger.info(f"Loading trained data: {trained_data_path}")
        adata = scv.read(trained_data_path)
        # print_anndata(adata)

        logger.info(f"Loading pyrovelocity data: {pyrovelocity_data_path}")
        posterior_samples = CompressedPickle.load(pyrovelocity_data_path)

        ##################
        # save dataframe
        ##################

        logger.info(f"Saving AnnData object to dataframe: {dataframe_path}")
        (
            df,
            total_obs,
            total_var,
            max_spliced,
            max_unspliced,
        ) = anndata_counts_to_df(adata)

        CompressedPickle.save(
            dataframe_path,
            (
                df,
                total_obs,
                total_var,
                max_spliced,
                max_unspliced,
            ),
        )
        print(posterior_samples.keys())

        vector_field_basis = data_model_conf.vector_field_parameters.basis
        # print(vector_field_basis)

        cell_type = data_model_conf.training_parameters.cell_state
        summarize_fig2_part1(
            adata,
            posterior_samples["vector_field_posterior_samples"],
            posterior_samples["cell_time"],
            posterior_samples["original_spaces_embeds_magnitude"],
            posterior_samples["pca_embeds_angle"],
            posterior_samples["embeds_angle"],
            vector_field_basis,
            posterior_samples["vector_field_posterior_mean"],
            cell_type,
            fig2_part1_plot,
        )

        summarize_fig2_part2(
            adata,
            posterior_samples,
            basis=vector_field_basis,
            cell_state=cell_type,
            plot_name=fig2_part2_plot,
            fig=None,
        )

        ##################
        # generate figures
        ##################

        # shared time plot
        cell_time_mean = posterior_samples["cell_time"].mean(0).flatten()
        cell_time_std = posterior_samples["cell_time"].std(0).flatten()
        adata.obs["shared_time_uncertain"] = cell_time_std
        adata.obs["shared_time_mean"] = cell_time_mean

        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(9.2, 2.6)
        ax = ax.flatten()
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_mean",
            ax=ax[0],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        ax_cb = scv.pl.scatter(
            adata,
            c="shared_time_uncertain",
            ax=ax[1],
            show=False,
            cmap="inferno",
            fontsize=7,
            colorbar=True,
        )
        select = adata.obs["shared_time_uncertain"] > np.quantile(
            adata.obs["shared_time_uncertain"], 0.9
        )
        sns.kdeplot(
            adata.obsm[f"X_{vector_field_basis}"][:, 0][select],
            adata.obsm[f"X_{vector_field_basis}"][:, 1][select],
            ax=ax[1],
            levels=3,
            fill=False,
        )
        ax[2].hist(cell_time_std / cell_time_mean, bins=100)
        fig.savefig(
            shared_time_plot,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            edgecolor="none",
            dpi=300,
        )

        # volcano plot
        if os.path.isfile(volcano_plot):
            logger.info(f"{volcano_plot} exists")
        else:
            logger.info(f"Generating figure: {volcano_plot}")
            fig, ax = plt.subplots()

            volcano_data, _ = plot_gene_ranking(
                [posterior_samples], [adata], ax=ax, time_correlation_with="st"
            )

            fig.savefig(
                volcano_plot,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                edgecolor="none",
                dpi=300,
            )
            print(
                volcano_data.sort_values("mean_mae", ascending=False)
                .head(300)
                .sort_values("time_correlation", ascending=False)
                .head(8)
            )

        # rainbow plot

        if os.path.isfile(rainbow_plot):
            logger.info(f"{rainbow_plot} exists")
        else:
            logger.info(f"Generating figure: {rainbow_plot}")
            fig = us_rainbowplot(
                volcano_data.sort_values("mean_mae", ascending=False)
                .head(300)
                .sort_values("time_correlation", ascending=False)
                .head(5)
                .index,
                adata,
                posterior_samples,
                data=["st", "ut"],
                cell_state=cell_state,
            )
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{rainbow_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )

        # mean vector field plot

        if os.path.isfile(vector_field_plot):
            logger.info(f"{vector_field_plot} exists")
        else:
            logger.info(f"Generating figure: {vector_field_plot}")
            fig, ax = plt.subplots()

            # embed_mean = plot_mean_vector_field(posterior_samples, adata, ax=ax)
            scv.pl.velocity_embedding_grid(
                adata,
                basis=vector_field_basis,
                color=cell_state,
                title="",
                vkey="velocity_pyro",
                linewidth=1,
                ax=ax,
                show=False,
                legend_loc="right margin",
                density=0.4,
                scale=0.2,
                arrow_size=2,
                arrow_length=2,
                arrow_color="black",
            )
            for ext in ["", ".png"]:
                fig.savefig(
                    f"{vector_field_plot}{ext}",
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    edgecolor="none",
                    dpi=300,
                )


@hydra.main(version_base="1.2", config_path=".", config_name="config.yaml")
def main(conf: DictConfig) -> None:
    """Plots figures for model summary.
    Args:
        config {DictConfig}: hydra configuration
    """

    logger = get_pylogger(name="PLOT", log_level=conf.base.log_level)
    plots(conf, logger)


if __name__ == "__main__":
    main()
