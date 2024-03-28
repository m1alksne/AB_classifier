# Use an official Jupyter image as a parent image
# More info: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM quay.io/jupyter/minimal-notebook:latest

# permissions
USER root
# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opensoundscape

# permissions
USER jovyan

# Clone git repo
# RUN git clone https://github.com/sergeygurvich/AB_classifier

# Run JupyterLab when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]