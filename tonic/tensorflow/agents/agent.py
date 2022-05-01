import random

import numpy as np
import tensorflow as tf

from tonic import agents, logger
physical_devices = tf.config.list_physical_devices('GPU')
import torch
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
        self.model.save_weights(path)
        path = path + '.pt'
        self.save_observation_normalizer(path)
        self.save_optimizer(path)

    def load(self, path, play=None):
        logger.log(f'\nLoading weights from {path}')
        from pudb import set_trace
        set_trace()
        self.model.load_weights(path)
        #self.load_observation_normalizer(path)


    def save_return_normalizer(self, path):
        if hasattr(self.model, 'return_normalizer'):
            reno = self.model.return_normalizer
            norm_path = self.get_path(path, 'ret_norm')
            ret_norm_dict = {'min_rew': reno.min_reward,
                             'max_rew': reno.max_reward,
                             'low' : reno._low,
                             'high': reno._high,
                             'coefficient': reno.coefficient}
            torch.save(ret_norm_dict, norm_path)

    def save_observation_normalizer(self, path):
        if hasattr(self.model, 'observation_normalizer'):
            ono = self.model.observation_normalizer
            norm_path = self.get_path(path, 'obs_norm')
            obs_norm_dict = {'clip': ono.clip,
                             'count': ono.count,
                             'mean' : ono.mean,
                             'mean_sq': ono.mean_sq,
                             'std': ono.std,
                             '_mean': ono._mean,
                             '_std': ono._std}
            torch.save(obs_norm_dict, norm_path)

    def load_observation_normalizer(self, path):
        if hasattr(self.model, 'observation_normalizer'):
            norm_path = self.get_path(path, 'obs_norm')
            load_dict = torch.load(norm_path)
            for k, v in load_dict.items():
                setattr(self.model.observation_normalizer, k, v)

    def load_return_normalizer(self, path):
        if hasattr(self.model, 'return_normalizer'):
            norm_path = self.get_path(path, 'ret_norm')
            load_dict = torch.load(norm_path)
            for k, v in load_dict.items():
                setattr(self.model.return_normalizer, k, v)

    def save_optimizer(self, path):
        if hasattr(self, 'actor_updater'):
            if hasattr(self.actor_updater, 'optimizer'):
                opt_path = self.get_path(path, 'actor')
                from pudb import set_trace
                set_trace()
                torch.save(self.actor_updater.optimizer.state_dict(), opt_path)
            else:
                # so far, only MPO has different optimizers
                opt_path = self.get_path(path, 'actor')
                torch.save(self.actor_updater.actor_optimizer.state_dict(), opt_path)
                opt_path = self.get_path(path, 'dual')
                torch.save(self.actor_updater.dual_optimizer.state_dict(), opt_path)
        if hasattr(self, 'critic_updater'):
            pass

    def get_path(self, path, post_fix):
        return path.split('step')[0] + post_fix + '.pt'
