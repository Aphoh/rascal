import sys
import os
import functools

import torch
import numpy as np
import torch_xla.distributed.spmd as xs
import torch_xla.utils.utils as xu
import torch_xla.distributed.parallel_loader as pl
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
from torch_xla import runtime as xr
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

import time
import itertools

import torch
import torch_xla
import torch.optim as optim
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import nn

from rascal.data.dummy import get_dummy_loader
from rascal.models.qwen import Qwen2Config, Qwen2DecoderLayer, Qwen2ForCausalLM


def apply_xla_flash_attention_with_spmd(query_states, key_states, value_states, attn_mask=None, **kwargs):
    from torch_xla.experimental.custom_kernel import flash_attention

    assert attn_mask is None
    head_dim = query_states.size()[-1]
    query_states = query_states / math.sqrt(head_dim)

    # Our simplified version of decoder only model does not use any mask.
    # flash_attention will use the global_mesh set in the TrainDecoderOnlyFSDPv2.
    attn_output = flash_attention(
        query_states, key_states, value_states, causal=True, partition_spec=('fsdp', None, None, None))
    return attn_output


# checkout our doc at https://github.com/pytorch/xla/blob/master/docs/fsdpv2.md
class TrainDecoderOnlyFSDPv2:

  def __init__(self):
    
    self.batch_size = 8
    self.seq_len = 4096
    self.num_steps = 200
    self.num_epochs = 1
    self.device = torch_xla.device()
    self.config = Qwen2Config(max_position_embeddings=self.seq_len)
    self.model = Qwen2ForCausalLM(self.config).to_bf16().to(self.device)
    for layer in self.model.model.layers:
        layer.self_attn.attn_impl = apply_xla_flash_attention_with_spmd

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    self.loss_fn = nn.CrossEntropyLoss()
    # Compile the step fn
    self.compiled_step_fn = torch_xla.compile(
        self.step_fn, full_graph=True, name="decoder_step_fn")

    # Define the mesh following common SPMD practice
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    xs.set_global_mesh(mesh)

    # Shard the input(data parallel).
    # Scale the batch size with num_devices since there will be only one
    # process that handles all runtime devices.
    self.batch_size *= num_devices
    train_loader = get_dummy_loader(128*self.batch_size, 1000, self.seq_len, 0, self.batch_size)
    self.train_device_loader = pl.MpDeviceLoader(
        train_loader,
        self.device,
        # Shard the input's batch dimension along the `fsdp` axis, no sharding along other dimensions
        input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)))

    # Apply FSDP sharding on each DecoderLayer layer.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2DecoderLayer
        },
    )
    # FSDPv2 will use the global mesh set above
    self.model = FSDPv2(
        self.model, auto_wrap_policy=auto_wrap_policy)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

  def _train_update(self, step, loss, tracker, epoch):
    print(f'epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()}')

  def run_optimizer(self):
    self.optimizer.step()

  def step_fn(self, input_ids, pos):
    self.optimizer.zero_grad()
    logits = self.model(input_ids, pos)
    logits = logits[:, :-1].view(-1, self.config.vocab_size)
    targets = input_ids[:, 1:].view(-1)
    loss = self.loss_fn(logits, targets)
    loss.backward()
    self.run_optimizer()
    return loss

  def train_loop_fn(self, loader, epoch):
    tracker = xm.RateTracker()
    self.model.train()
    loader = itertools.islice(loader, self.num_steps)
    inp_pos = torch.arange(self.seq_len, device=self.device)
    for step, data in enumerate(loader):
      loss = self.compiled_step_fn(data["sequences"], inp_pos)
      tracker.add(self.batch_size)
      if step % 10 == 0:
        xm.add_step_closure(
            self._train_update, args=(step, loss, tracker, epoch))

  def start_training(self):

    pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params/1e6:.2f}M")


    for epoch in range(1, self.num_epochs + 1):
      xm.master_print('Epoch {} train begin {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
      self.train_loop_fn(self.train_device_loader, epoch)
      xm.master_print('Epoch {} train end {}'.format(
          epoch, time.strftime('%l:%M%p %Z on %b %d, %Y')))
    xm.wait_device_ops()


if __name__ == '__main__':
  # Enable the SPMD
  xr.initialize_cache('.hlo_cache', readonly=False)
  xr.use_spmd()
  base = TrainDecoderOnlyFSDPv2()
  base.start_training()

