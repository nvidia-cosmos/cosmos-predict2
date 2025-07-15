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

# ------------------------------------------------------------------
# Install latest Flash-Attention (Blackwell) without the NGC pin
# ------------------------------------------------------------------
RUN pip uninstall -y flash-attn || true && \
    printf '' > /tmp/empty-constraints.txt && \
    PIP_CONSTRAINT=/tmp/empty-constraints.txt \
    MAX_JOBS=$(( $(nproc) / 4 )) \
    pip install --no-deps git+https://github.com/Dao-AILab/flash-attention.git@main

# Copy the entire cosmos_predict2 directory for conditional file operations
COPY . /tmp/cosmos_build/

# Conditionally copy Flash Attention interface files if they exist
RUN if [ -f /tmp/cosmos_build/cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py ]; then \
        cp /tmp/cosmos_build/cosmos_predict2/utils/flash_attn_3/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/flash_attn_3/flash_attn_interface.py; \
        echo "Copied Flash Attention interface"; \
    else \
        echo "Flash Attention interface file not found, skipping"; \
    fi

# Conditionally apply Transformer Engine patch if available
RUN if [ -f /tmp/cosmos_build/cosmos_predict2/utils/flash_attn_3/te_attn.diff ]; then \
        cp /tmp/cosmos_build/cosmos_predict2/utils/flash_attn_3/te_attn.diff /tmp/te_attn.diff && \
        patch /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/attention.py /tmp/te_attn.diff && \
        echo "Applied Transformer Engine patch"; \
    else \
        echo "Transformer Engine patch not found, skipping"; \
    fi

# Clean up temporary build files
RUN rm -rf /tmp/cosmos_build

RUN mkdir -p /workspace
WORKDIR /workspace

CMD ["/bin/bash"]
