# Use an official Jupyter image as a parent image
# More info: https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
FROM quay.io/jupyter/minimal-notebook:latest

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opensoundscape

# Clone git repo
# RUN git clone https://github.com/Dr-Nathan-Fox/EcoViz_CLIP

# Run JupyterLab when the container launches
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]