# Minimal environment for the NeRF viewer OpenGL window (Ubuntu 24.04, software rendering).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    python3 python3-pip python3-jinja2 \
    libglm-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    mesa-utils xauth \
    && rm -rf /var/lib/apt/lists/*

# GLAD 1 generator (local in ./glad) and the GL 3.3 core loader pre-generated in /opt/glad_out
COPY glad /opt/glad
RUN python3 -m pip install --break-system-packages /opt/glad \
    && glad --profile core --api gl=3.3 --generator c --out-path /opt/glad_out

WORKDIR /workspace
CMD ["/bin/bash"]
