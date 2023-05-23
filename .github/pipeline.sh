#!/usr/bin/env bash

#########################
# To test uncommitted changes and/or individual pipeline stages
# replace `dvc repro` with something analogous to:
#     dvc exp run \
#     -S config.yaml:model_training.pancreas_model1.training_parameters.max_epochs=421 \
#     train_pancreas_model1
# TODO: remove npm update ... after container rebuild
# DONE: replace npx ... with cml
# npx github:iterative/cml#b923f3e pr create --skip-ci .
# https://github.com/iterative/cml/issues/1344 resolved in
# https://github.com/iterative/cml/releases/tag/v0.18.21
# TODO: move matplotlib-venn to package manager if persistent
# TODO: move /usr/bin/time install to Docker
#########################

set -x


### Define parallel execution function ###
function run_parallel_pipeline() {
    dvc stage list --name-only |\
        grep -E "data_download*" |\
        xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 8 + 3)); dvc repro "$@"' --

    wait

    dvc stage list --name-only |\
        grep -E "preprocess*" |\
        xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 8 + 3)); dvc repro "$@"' --

    wait

    # manually execute training stages to distribute over four GPUs
    dvc repro train@pancreas_model2 &
    sleep 2
    dvc repro train@pbmc68k_model2 &
    sleep 2
    dvc repro train@pons_model2 &
    sleep 2
    dvc repro train@larry_model2 &
    wait
    dvc repro train@larry_tips_model2 &
    sleep 2
    dvc repro train@larry_mono_model2 &
    sleep 2
    dvc repro train@larry_neu_model2 &
    sleep 2
    dvc repro train@larry_multilineage_model2 &
    wait
    dvc repro train@pbmc10k_model2 &

    wait

    dvc stage list --name-only |\
        grep -E "postprocess*" |\
        xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 8 + 3)); dvc repro "$@"' --

    wait

    dvc stage list --name-only |\
        grep -E "summarize*" |\
        xargs -t -n 1 -P 4 bash -c 'sleep $((RANDOM % 8 + 3)); dvc repro "$@"' --

    wait
}


### Execute experiment run and submit PR ###
python --version
pip --version
pip list
sudo apt-get update && sudo apt-get install -y time && which time
cd reproducibility/figures || exit

dvc pull
run_parallel_pipeline
dvc repro
dvc push

npm update -g @dvcorg/cml
cml pr create --skip-ci .
#########################

generate_markdown() {
    local data_set="$1"
    local model="$2"

    printf "\n# %s\n\n## %s\n\n### Metrics\n" "$data_set" "$model"
    dvc metrics show --md "models/${data_set}_${model}/metrics.json"

    echo "### Data summary"
    echo '!'"[Raw Count Histogram](./data/processed/${data_set}_thresh_histogram.pdf.png)"

    echo "### Training plots"
    echo '!'"[ELBO](./models/${data_set}_${model}/loss_plot.png)"

    echo "### Run information"
    printf "\n\`\`\`json\n"
    cat "models/${data_set}_${model}/run_info.json"
    printf "\n\`\`\`\n"
}

### Post experiment report comment ###
{
printf "# pipeline\n"
dvc dag --md

data_sets=(
    "pancreas"
    "pons"
    "pbmc68k"
    "pbmc10k"
    "larry"
    "larry_mono"
    "larry_neu"
    "larry_multilineage"
    "larry_tips"
)
models=("model2")

for data_set in "${data_sets[@]}"; do
    for model in "${models[@]}"; do
        generate_markdown "$data_set" "$model"
    done
done

printf "# pipeline files\n"
dvc dag -o --md
} >> report.md

cml comment update report.md
#########################

set +x
