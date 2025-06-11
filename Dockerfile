# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Install basic tools
RUN apt-get update && apt-get install -y git tree ffmpeg wget
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && ln -s /lib64/libcuda.so.1 /lib64/libcuda.so
RUN apt-get install -y libglib2.0-0
RUN sed -i -e 's/h11==0.14.0/h11==0.16.0/g' /etc/pip/constraint.txt

# Install the dependencies from requirements-docker.txt
COPY ./requirements-docker.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
# Install Flash Attention 3
RUN MAX_JOBS=$(( $(nproc) / 4 )) pip install git+https://github.com/Dao-AILab/flash-attention.git@27f501d#subdirectory=hopper
COPY cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/flash_attn_interface.py
COPY cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff
RUN patch /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff

RUN mkdir -p /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
