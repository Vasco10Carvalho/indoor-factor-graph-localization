FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        python3-pip \
        ros-humble-geometry-msgs \
        ros-humble-nav-msgs \
        ros-humble-rosbag2 \
        ros-humble-rosbag2-storage-mcap \
        ros-humble-rviz2 \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
        ros-humble-tf2-ros \
        ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade "pip<26" "setuptools<80" wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /workspace/pipeline.sh /workspace/indoor_localization/*.py \
    && python3 -m py_compile /workspace/indoor_localization/*.py \
    && source /opt/ros/humble/setup.bash \
    && python3 -m unittest discover -s /workspace/tests -p 'test_*.py'

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
