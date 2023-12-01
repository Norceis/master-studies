FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y \
    git  \
    nano
    
COPY 3d-pokemon-generator  3d-pokemon-generator
RUN cd 3d-pokemon-generator/singleshapegen && \
	pip install -r requirements.txt

