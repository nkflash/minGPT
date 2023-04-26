"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mingpt.model import configure_optimizers
from torch.distributed.checkpoint import (
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
    FileSystemReader,
)




class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.launch_type = 'local'
        C.checkpoint_iters = None
        C.start_from_checkpoint = False
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        if self.config.launch_type != 'local':
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda:' + str(self.local_rank) if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print(f'running on device {self.device} global_rank:{self.global_rank} local_rank:{self.local_rank}')

        if self.config.launch_type == 'ddp':
            self.model = DDP(self.model, device_ids=[self.local_rank])

        self.iter_num = 0
        self.dir_locate = 'checkpoint/' + str(self.global_rank)
        if config.start_from_checkpoint:
            sd = {'module': model.state_dict(), 'iteration': 0}
            fs_storage_reader = FileSystemReader(self.dir_locate)
            load_state_dict(
                state_dict=sd,
                storage_reader=fs_storage_reader,
            )
            print(f'start train from checkpoint, iterator:{sd["iteration"]}')
            self.iter_num = sd["iteration"]
            model.load_state_dict(sd["module"])
        # variables that will be assigned to trainer class later for logging and etc
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = configure_optimizers(model, config)

        if config.launch_type != 'local':
            sampler = DistributedSampler(self.train_dataset)
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10))

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config.checkpoint_iters is not None and self.iter_num % config.checkpoint_iters == 0:
                model_state_dict = model.state_dict()
                sd = {'module': model_state_dict, 'iteration': self.iter_num}
                fs_storage_writer = FileSystemWriter(self.dir_locate)
                save_state_dict(
                    state_dict=sd,
                    storage_writer=fs_storage_writer,
                )

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
