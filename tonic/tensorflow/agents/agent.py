import random
import torch
import numpy as np
import tensorflow as tf
import os
from tonic import agents, logger
physical_devices = tf.config.list_physical_devices('GPU')
try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
      # Invalid device or cannot modify virtual devices once initialized.
        pass



class Agent(agents.Agent):
    def initialize(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

    def save(self, path):
        logger.log(f'\nSaving weights to {path}')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        logger.log(f'\nSaving weights to {path}')
        self.model.save_weights(path)
        path = path + '.pt'
        self.save_return_normalizer(path)
        self.save_observation_normalizer(path)
        self.save_buffer(path)
        self.save_optimizer(path)

    def load(self, path, play=None):
        logger.log(f'\nLoading weights from {path}')
        self.load_optimizer(path)
        self.model.load_weights(path)
        # self.load_buffer(path)
        self.load_observation_normalizer(path)
        self.load_return_normalizer(path)
        self.load_buffer(torch.load, path)

    def save_return_normalizer(self, path):
        if self.model.return_normalizer is not None:
            reno = self.model.return_normalizer
            norm_path = self.get_path(path, 'ret_norm')
            ret_norm_dict = {'min_rew': reno.min_reward,
                             'max_rew': reno.max_reward,
                             '_low' : reno._low,
                             '_high': reno._high,
                             'coefficient': reno.coefficient}
            torch.save(ret_norm_dict, norm_path)

    def save_observation_normalizer(self, path):
        if self.model.observation_normalizer is not None:
            ono = self.model.observation_normalizer
            norm_path = self.get_path(path, 'obs_norm')
            obs_norm_dict = {'clip': ono.clip,
                             'count': ono.count,
                             'mean': ono.mean,
                             'mean_sq': ono.mean_sq,
                             'std': ono.std,
                             '_mean': ono._mean,
                             '_std': ono._std}
            torch.save(obs_norm_dict, norm_path)

    def load_observation_normalizer(self, path):
        if self.model.observation_normalizer is not None:
            norm_path = self.get_path(path, 'obs_norm')
            load_dict = torch.load(norm_path)
            for k, v in load_dict.items():
                setattr(self.model.observation_normalizer, k, v)

    def load_return_normalizer(self, path):
        if self.model.return_normalizer is not None:
            norm_path = self.get_path(path, 'ret_norm')
            load_dict = torch.load(norm_path)
            for k, v in load_dict.items():
                setattr(self.model.observation_normalizer, k, v)

    def save_buffer(self, path):
        self.replay.save(path)

    def load_buffer(self, load_fn, path):
        self.replay.load(load_fn, path)

    def save_optimizer(self, path):
        for updater in ['actor_updater', 'critic_updater']:
            if hasattr(self, updater):
                if hasattr(getattr(self, updater), 'optimizer'):
                    opt = getattr(self, updater).optimizer
                    opt_path = self.get_path(path, updater)
                    torch.save(opt.get_weights(), opt_path)

    def load_optimizer(self, path):
        for updater in ['actor_updater', 'critic_updater']:
            if hasattr(self, updater):
                if hasattr(getattr(self, updater), 'optimizer'):
                    opt = getattr(self, updater).optimizer
                    opt_path = self.get_path(path, updater)
                    load_dict = torch.load(opt_path)
                    if 'actor' in updater:
                        grad_vars = self.actor_updater.model.actor.trainable_variables
                    else:
                        grad_vars = self.critic_updater.model.critic_1.trainable_variables + self.critic_updater.model.critic_2.trainable_variables
                    zero_grads = [tf.zeros_like(w) for w in grad_vars]
                    opt.apply_gradients(zip(zero_grads, grad_vars))
                    opt.set_weights(load_dict)

    def get_path(self, path, post_fix):
        return path.split('step')[0] + post_fix + '.pt'
