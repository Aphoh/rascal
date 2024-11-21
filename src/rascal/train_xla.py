from rascal.models.qwen import Qwen2ForCausalLM, Qwen2Config

from torch_xla import runtime as xr
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr

import time
import itertools
from .data.dummy import get_dummy_loader
from .models.router import RoutedLoraLLM

import torch
import torch_xla
import torch.optim as optim
import torch.nn as nn

xr.initialize_cache(".hlo_cache", readonly=False)


class TrainDecoderOnlyBase:

    def __init__(self):
        self.config = Qwen2Config()
        self.batch_size = 16
        self.seq_len = self.config.max_position_embeddings
        self.num_steps = 200
        self.num_epochs = 1
        self.train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
        # For the purpose of this example, we are going to use fake data.
        train_loader = get_dummy_loader(
            self.train_dataset_len,
            self.config.vocab_size,
            self.config.max_position_embeddings,
            self.config.pad_token_id,
            self.batch_size,
        )

        self.device = torch_xla.device()
        self.train_device_loader = pl.MpDeviceLoader(train_loader, self.device)
        self.llm = Qwen2ForCausalLM(self.config)
        self.model = RoutedLoraLLM(self.llm, self.config.hidden_size, 64, 8)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        # Compile the step fn
        self.compiled_step_fn = torch_xla.compile(
            self.step_fn, full_graph=True, name="decoder_step_fn"
        )
        self.compiled_suffix_hiddens_fn = torch_xla.compile(
            self.get_suffix_hiddens, full_graph=True, name="suffix_hiddens_fn"
        )

    def _train_update(self, step, loss, tracker, epoch, n_tokens):
        print(f"epoch: {epoch}, step: {step}, loss: {loss}, rate: {tracker.rate()} n_tokens: {n_tokens}")

    def run_optimizer(self):
        self.optimizer.step()

    def step_fn(self, input_ids, input_positions, suffix_idx, lens):
        self.optimizer.zero_grad()
        loss, n_tokens = self.model.compute_loss({"input_ids": input_ids, "input_positions": input_positions}, suffix_idx, lens)
        loss.backward()
        self.run_optimizer()
        return loss, n_tokens

    def train_loop_fn(self, loader, epoch):
        tracker = xm.RateTracker()
        n_tokens = 0
        self.model.train()
        loader = itertools.islice(loader, self.num_steps)
        for step, data in enumerate(loader):
            input_ids = data["sequence"]
            lens = data["length"]
            suffix_idxs = data["suffix_idx"]
            input_positions = torch.arange(0, self.seq_len, device=self.device)
            loss, step_n_tokens = self.compiled_step_fn(
                input_ids, input_positions, suffix_idxs, lens
            )
            tracker.add(self.batch_size)
            n_tokens += step_n_tokens
            if step % 10 == 0:
                xm.add_step_closure(
                    self._train_update, args=(step, loss, tracker, epoch, n_tokens)
                )

    def start_training(self):

        for epoch in range(1, self.num_epochs + 1):
            xm.master_print(
                "Epoch {} train begin {}".format(
                    epoch, time.strftime("%l:%M%p %Z on %b %d, %Y")
                )
            )
            self.train_loop_fn(self.train_device_loader, epoch)
            xm.master_print(
                "Epoch {} train end {}".format(
                    epoch, time.strftime("%l:%M%p %Z on %b %d, %Y")
                )
            )
        xm.wait_device_ops()


if __name__ == "__main__":
    base = TrainDecoderOnlyBase()
    base.start_training()
