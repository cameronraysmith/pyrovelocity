# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:23.3.1-1


WORKDIR /pyrovelocity
COPY . .

RUN mamba init
RUN mamba env update -n base -f app/environment.yml
RUN pip install --no-deps -e .
RUN dvc stage list \
    --name-only reproducibility/figures/dvc.yaml | \
    grep -E "summarize" | \
    xargs -t -I {} dvc pull {}

LABEL org.opencontainers.image.title="pyrovelocityapp" \
      org.opencontainers.image.authors="pyrovelocity team" \
      org.opencontainers.image.description="This image contains the pyrovelocity web application." \
      org.opencontainers.image.url="https://github.com/pinellolab/pyrovelocity" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

CMD streamlit run app/app.py \
	--server.port=8080 \
	--server.enableCORS=false
