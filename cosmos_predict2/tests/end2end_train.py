# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
test end2end training
Usage:
    * [run all tests]: torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2/tests/end2end_train.py --L1
"""

import importlib
import os
import shutil

import pytest
import torch

from cosmos_predict2.models.wan_imaginaire_model import WanImaginaireModel
from imaginaire.lazy_config import instantiate
from imaginaire.utils import distributed, log
from imaginaire.utils.config_helper import get_config_module, override
from imaginaire.utils.helper_test import RunIf


@RunIf(min_gpus=8)
@pytest.mark.L1
def test_fsdp_wan_test_ckpt():
    """
    This test simulate the process to verify the checkpointer is working as expected.
    1. Init model, compute loss_0
    2. Save checkpoint
    2. Update model weights, compute loss_1 (expected different from loss_0)
    3. Load checkpoint, compute loss_3 (expected the same as loss_0)

    torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2.tests.end2end_train::test_fsdp_wan_test_ckpt --L1
    """
    distributed.init()

    config_file = "cosmos_predict2/configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            "experiment=wan_exp000_001_14b_i2v_480p_fsdp_lora",
            "data_train=groot_local_single_sample",
            "model.config.debug_without_randomness=True",
            "job.group=debug",
            "job.name=fsdp_wan_ckpt_save_unit_test_1p3b",
            "model=wan_t2v_1p3b_fsdp_lora",
        ],
    )
    # Init trainer
    trainer = config.trainer.type(config)
    # Init model
    model: WanImaginaireModel = instantiate(config.model)
    model.on_model_init_end()
    model = model.to("cuda").to(torch.bfloat16)

    # Initialize the optimizer and scheduler, grad_scaler
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)

    dataloader_train = instantiate(config.dataloader_train)
    # Forward the model to get the output
    for batch in dataloader_train:
        output, loss_0 = model.training_step(batch, 0)
        break
    expected_loss_1p3b_t2v_model = 0.027800431475043297
    assert (
        loss_0.item() == expected_loss_1p3b_t2v_model
    ), f"expected loss_0 = {expected_loss_1p3b_t2v_model}, but got {loss_0.item()}"
    # Delete checkpoint path if exists
    resume_keys, checkpoint_path = trainer.checkpointer.keys_to_resume_during_load()
    if checkpoint_path is not None and os.path.exists(checkpoint_path) and distributed.get_rank() == 0:
        # remove the folder
        log.info(f"Removing checkpoint path {checkpoint_path}")
        shutil.rmtree(checkpoint_path)
    # Save first checkpoint
    trainer.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=0)

    # Update the model weights
    loss_0.backward()
    optimizer.step()

    # Forward again to get new loss
    with torch.no_grad():
        for batch in dataloader_train:
            output, loss_1 = model.training_step(batch, 1)
            break
        assert (
            loss_1.item() != loss_0.item()
        ), f"expected loss_1 become different from loss_0, but got {loss_1.item()} and {loss_0.item()}"

        # Forward again to check if the loss is the same
        for batch in dataloader_train:
            output, loss_2 = model.training_step(batch, 1)
            break
        assert loss_2.item() == loss_1.item(), f"expected loss_2 = loss_1, but got {loss_2.item()} and {loss_1.item()}"

    # re-init model and load the checkpoint, verify the loss is the same as the original loss
    trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    # with torch.no_grad():
    for batch in dataloader_train:
        output, loss_3 = model.training_step(batch, 1)
        break
    assert loss_3.item() == loss_0.item(), f"expected loss_3 = loss_0, but got {loss_3.item()} and {loss_0.item()}"

    # Delete checkpoint path if exists
    resume_keys, checkpoint_path = trainer.checkpointer.keys_to_resume_during_load()
    if checkpoint_path is not None and os.path.exists(checkpoint_path) and distributed.get_rank() == 0:
        # remove the folder
        log.info(f"Removing checkpoint path {checkpoint_path}")
        shutil.rmtree(checkpoint_path)


@RunIf(min_gpus=8)
@pytest.mark.L1
def test_fsdp_wan_test_forward():
    """
    This test verify the loss of init model matches the expected loss.
    torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2.tests.end2end_train::test_fsdp_wan_test_forward --L1
    """
    distributed.init()

    config_file = "cosmos_predict2/configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            "experiment=wan_exp000_001_14b_i2v_480p_fsdp_lora",
            "data_train=groot_local_single_sample",
            "model.config.debug_without_randomness=True",
        ],
    )
    model: WanImaginaireModel = instantiate(config.model)
    model.on_model_init_end()
    model = model.to("cuda").to(torch.bfloat16)

    dataloader_train = instantiate(config.dataloader_train)
    # Forward the model to get the output
    for batch in dataloader_train:
        output, loss = model.training_step(batch, 0)
        break
    # Check if loss is the expected value when using debug_without_randomness mode
    expected_loss_14b_i2v_480p_model = 0.011205743066966534
    assert (
        loss.item() == expected_loss_14b_i2v_480p_model
    ), f"expected loss = {expected_loss_14b_i2v_480p_model}, but got {loss.item()}"


@RunIf(min_gpus=8)
@pytest.mark.L1
def test_ddp_wan_test_forward():
    """
    torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2.tests.end2end_train::test_ddp_wan_test_forward --L1
    """
    distributed.init()

    config_file = "cosmos_predict2/configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            "experiment=wan_exp000_000_14b_i2v_480p_ddp_lora",
            "data_train=groot_local_single_sample",
            "model.config.debug_without_randomness=True",
        ],
    )
    model: WanImaginaireModel = instantiate(config.model)
    model.on_model_init_end()
    model = model.to("cuda").to(torch.bfloat16)

    dataloader_train = instantiate(config.dataloader_train)
    # Forward the model to get the output
    for batch in dataloader_train:
        output, loss = model.training_step(batch, 0)
        break
    # Check if loss is the expected value when using debug_without_randomness mode
    expected_loss_14b_i2v_480p_model = 0.011205743066966534
    assert (
        loss.item() == expected_loss_14b_i2v_480p_model
    ), f"expected loss = {expected_loss_14b_i2v_480p_model}, but got {loss.item()}"


@RunIf(min_gpus=8)
@pytest.mark.L1
def test_ddp_wan_1_3B_test_forward():
    """
    torchrun --nproc_per_node=8 -m pytest -s cosmos_predict2.tests.end2end_train::test_ddp_wan_1_3B_test_forward --L1
    """
    distributed.init()

    config_file = "cosmos_predict2/configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            "experiment=wan_exp000_004_1_3b_t2v_ddp_lora",
            "data_train=groot_local_single_sample",
            "model.config.debug_without_randomness=True",
        ],
    )
    model: WanImaginaireModel = instantiate(config.model)
    model.on_model_init_end()
    model = model.to("cuda").to(torch.bfloat16)

    dataloader_train = instantiate(config.dataloader_train)
    # Forward the model to get the output
    for batch in dataloader_train:
        output, loss = model.training_step(batch, 0)
        break
    # Check if loss is the expected value when using debug_without_randomness mode
    expected_loss_1_3b_t2v_model = 0.027800431475043297
    assert (
        loss.item() == expected_loss_1_3b_t2v_model
    ), f"expected loss = {expected_loss_1_3b_t2v_model}, but got {loss.item()}"
