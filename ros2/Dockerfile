# Multi-stage Dockerfile for ROS2 RAG Service
# Based on ROS2 Humble with Python 3.10

FROM osrf/ros:humble-desktop-full AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=humble

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the RAG system
RUN pip3 install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-openai \
    langchain-chroma \
    langchain-text-splitters \
    chromadb \
    pypdf \
    python-dotenv \
    redis \
    rank_bm25 \
    pydantic \
    tiktoken
    # OLLAMA PACKAGES (COMMENTED OUT - NOW USING CHATGPT/OPENAI):
    # langchain-ollama \
    # ollama

# Create workspace
WORKDIR /ros2_ws

# Copy the ROS2 packages
COPY rag_interfaces /ros2_ws/src/rag_interfaces
COPY rag_service /ros2_ws/src/rag_service

# Source ROS2 and build the packages (interfaces first, then service)
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /ros2_ws && \
    colcon build --packages-select rag_interfaces --event-handlers console_direct+ && \
    source /ros2_ws/install/setup.bash && \
    colcon build --packages-select rag_service --event-handlers console_direct+"

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Source ROS2 setup\n\
source /opt/ros/${ROS_DISTRO}/setup.bash\n\
source /ros2_ws/install/setup.bash\n\
\n\
# Execute the command\n\
exec "$@"' > /ros2_entrypoint.sh && \
    chmod +x /ros2_entrypoint.sh

ENTRYPOINT ["/ros2_entrypoint.sh"]
CMD ["ros2", "launch", "rag_service", "rag_service_launch.py"]
