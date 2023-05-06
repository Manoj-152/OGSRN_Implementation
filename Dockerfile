# Official Docker python image
# FROM alpine:latest
FROM python:3.7

# Setup basic devel 
RUN apt-get update

# Setup working directory
RUN mkdir -p /SuperResolutionV1
WORKDIR /SuperResolutionV1

# Copy and add files to workdir
ADD Preprocessing /SuperResolutionV1
ADD Inference_Input /SuperResolutionV1
ADD Inference_Output /SuperResolutionV1
ADD Runner /SuperResolutionV1
ADD Models /SuperResolutionV1

COPY requirements.txt /SuperResolutionV1
COPY config.yaml /SuperResolutionV1
COPY main.py /SuperResolutionV1
COPY SRUN_best_ckpt.pth /SuperResolutionV1
COPY SORTN_best_ckpt.pth /SuperResolutionV1
COPY run_script.sh /SuperResolutionV1

# Install dependencies
RUN pip3 install -r /SuperResolutionV1/requirements.txt

# Make the script executable
RUN chmod +x /SuperResolutionV1/run_script.sh

# Start Sub-processes
CMD ./SuperResolutionV1/run_script.sh inference

# Docker Commands
# See all images
# docker images
# Build image
# docker build -t SuperResolutionV1:1
# Run image
# docker run -d -ti --rm --name=SuperResolutionV1 SuperResolutionV1:1
# View docker running processes
# docker ps
# Get Interactive Terminal inside container
# docker run -it SuperResolutionV1:1 /bin/bash