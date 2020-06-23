# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

# Now let's download python 3 and all the dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libav-tools \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
COPY ./files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Not sure why this is needed
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### MuJoCo
##########################################################
# Note: ~ is an alias for /root
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./files/mjkey.txt /root/.mujoco/mjkey.txt
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}



##########################################################
### Python
##########################################################
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y

RUN conda create --name railrl python=3.5.6 pip
RUN echo "source activate railrl" >> ~/.bashrc
# Use the railrl pip
ENV OLDPATH $PATH
ENV PATH /opt/conda/envs/railrl/bin:$PATH

RUN pip install imagehash>=3.4
RUN pip install ipdb
RUN pip install Pillow>=4.0.0
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail==0.3.0
RUN pip install scipy>=0.18.0
RUN pip install glfw>=1.4.0
RUN pip install imageio>=2.1.2
RUN pip install opencv-python==3.4.0.12
RUN pip install sk-video==1.1.10
RUN pip install numpy-stl==2.7.0
RUN pip install pyquaternion==0.9.2
RUN pip install moviepy==0.2.3.5
RUN pip install scikit-image
RUN pip install gitpython==2.1.7
RUN pip install gtimer==1.0.0b5
RUN pip install joblib==0.9.4
RUN pip install dominate==2.3.1
RUN pip install path.py==10.3.1
RUN pip install cached-property==1.3.1
RUN pip install cloudpickle==1.3.0
RUN pip install matplotlib==2.2.2
RUN pip install pygame==1.9.6
RUN pip install awscli==1.11.179
RUN pip install boto3==1.4.8
RUN pip install python-dateutil==2.6.1
RUN pip install torch==1.1.0
RUN pip install torchvision

########
### Mujoco-py and gym
########
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN pip install gym[all]==0.17.1
#RUN pip install pyqt5
RUN conda install -c https://conda.binstar.org/sstromberg pydot
RUN conda install GraphViz
RUN pip install pydot-ng
RUN pip install git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano
RUN pip install pyopengl
RUN pip install mako
RUN pip install --upgrade pip
RUN pip install pyqt5
##########################################################
### Cleanup
##########################################################
# I'm going to keep PATH to prioritize the conda env
# ENV PATH $OLDPATH
