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

    def load(self, path):
        path = path + '.pt'
        logger.log(f'\nLoading weights from {path}')
        if not torch.cuda.is_available():
            load_fn = partial(torch.load, map_location='cpu')
        else:
            load_fn = torch.load
        self.model.load_state_dict(load_fn(path))
        self.load_optimizer(load_fn, path)

    def save_optimizer(self, path):
        if hasattr(self, 'actor_updater'):
            actor_path = self.get_path(path, 'actor')
            torch.save(self.actor_updater.optimizer.state_dict(), actor_path)
        if hasattr(self, 'critic_updater'):
            critic_path = self.get_path(path, 'critic')
            torch.save(self.critic_updater.optimizer.state_dict(), critic_path)

    def load_optimizer(self, load_fn, path):
        if hasattr(self, 'actor_updater'):
            actor_path = self.get_path(path, 'actor')
            self.actor_updater.optimizer.load_state_dict(load_fn(actor_path))

        if hasattr(self, 'critic_updater'):
            critic_path = self.get_path(path, 'critic')
            self.critic_updater.optimizer.load_state_dict(load_fn(critic_path))

    def get_path(self, path, post_fix):
        return path.split('step')[0] + post_fix + '.pt'
