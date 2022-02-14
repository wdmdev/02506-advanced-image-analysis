FROM continuumio/miniconda3:4.10.3

# Build tools with C++ compiler for implicit library 
RUN apt-get update && apt-get install build-essential -y && apt-get install ffmpeg libsm6 libxext6 -y

#Install Python libraries
COPY environment.yml .
RUN conda env create -f environment.yml

# Activate conda for local development with orgenv kernel conda environment 
SHELL ["conda", "run", "-n", "adv-img", "/bin/bash", "-c"]