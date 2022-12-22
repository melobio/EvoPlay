# hash:sha256:55d1e6248cd63f399dfaee9909795353ea226adfbd79658b038f8691fb364159
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys \
        0xAD2323F17326AE31401037733E05EBFF05441C52 \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y 'deb http://deb.codeocean.com/rstudio-server-bionic/ ubuntu main' \
    && apt-get purge -y --autoremove software-properties-common

RUN conda install -y \
        numpy==1.23.4 \
        pandas==1.1.5 \
        python==3.8.5 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    scikit-learn==0.24.1 \
    tape-proteins==0.5 \
    torch==1.13.1 \
    typing-extensions==3.7.4.3 \
    xlrd==1.2.0

COPY postInstall /
RUN /postInstall
