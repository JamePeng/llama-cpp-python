# Define the base image version with CUDA 12.8 and cuDNN for Ubuntu 22.04
ARG CUDA_IMAGE="12.8.1-cudnn-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install system dependencies:
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y build-essential \
    ccache cmake curl gcc git wget \
    python3 python3-pip python3-wheel \
    libgomp1 libjpeg-dev libssl-dev libcurl4-openssl-dev

# Set the working directory for the container
WORKDIR /app

# Copy the current directory contents into the container (useful for local config files or models)
COPY . .

# Set the target GPU architecture (default allows CMake to auto-detect or use common archs)
ENV CUDA_DOCKER_ARCH=default
ENV GGML_CUDA=1
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# Enable verbose build output
ENV VERBOSE=1
# Specific CUDA paths to ensure the compiler (nvcc) is found
ENV CUDA_HOME="/usr/local/cuda/"
ENV CUDA_PATH="/usr/local/cuda/:${PATH}"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda/"

# Install Python build tools and server dependencies
RUN python3 -m pip install --upgrade pip pytest wheel packaging scikit-build setuptools fastapi uvicorn pydantic-settings sse-starlette starlette-context huggingface-hub

# Clone the source code. --recursive is CRITICAL to fetch the 'llama.cpp' submodule
RUN git clone --recursive https://github.com/JamePeng/llama-cpp-python.git

# Switch context to the cloned repository directory
WORKDIR /app/llama-cpp-python

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}" pip install .

# Run the server
CMD python3 -m llama_cpp.server
