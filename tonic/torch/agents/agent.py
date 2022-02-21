import os
import random
from functools import partial
import numpy as np
import torch

from tonic import agents, logger  # noqa



class Agent(agents.Agent):
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

    def save(self, path):
        path = path + '.pt'
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.save_optimizer(path)
        self.save_buffer(path)

    def load(self, path):
        path = path + '.pt'
        logger.log(f'\nLoading weights from {path}')
        if not torch.cuda.is_available():
            load_fn = partial(torch.load, map_location='cpu')
        else:
            load_fn = torch.load
        self.model.load_state_dict(load_fn(path))
        try:
            self.load_optimizer(load_fn, path)
            self.load_buffer(load_fn, path)
        except:
            print('Failure, only loading policy')

    def save_optimizer(self, path):
        if hasattr(self, 'actor_updater'):
            if hasattr(self.actor_updater, 'optimizer'):
                opt_path = self.get_path(path, 'actor')
                torch.save(self.actor_updater.optimizer.state_dict(), opt_path)
            else:
                # so far, only MPO has different optimizers
                opt_path = self.get_path(path, 'actor')
                torch.save(self.actor_updater.actor_optimizer.state_dict(), opt_path)
                opt_path = self.get_path(path, 'dual')
                torch.save(self.actor_updater.dual_optimizer.state_dict(), opt_path)
        if hasattr(self, 'critic_updater'):
            opt_path = self.get_path(path, 'critic')
            torch.save(self.critic_updater.optimizer.state_dict(), opt_path)

    def load_optimizer(self, load_fn, path):
        if hasattr(self, 'actor_updater'):
            if hasattr(self.actor_updater, 'optimizer'):
                opt_path = self.get_path(path, 'actor')
                self.actor_updater.optimizer.load_state_dict(load_fn(opt_path))
            else:
                opt_path = self.get_path(path, 'actor')
                self.actor_updater.actor_optimizer.load_state_dict(load_fn(opt_path))
                opt_path = self.get_path(path, 'dual')
                self.actor_updater.dual_optimizer.load_state_dict(load_fn(opt_path))

        if hasattr(self, 'critic_updater'):
            opt_path = self.get_path(path, 'critic')
            self.critic_updater.optimizer.load_state_dict(load_fn(opt_path))

    def save_buffer(self, path):
        self.replay.save(path)

    def load_buffer(self, load_fn, path):
        self.replay.load(load_fn, path)

    def get_path(self, path, post_fix):
        return path.split('step')[0] + post_fix + '.pt'
