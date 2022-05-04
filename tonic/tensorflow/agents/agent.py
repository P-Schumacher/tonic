import random

import numpy as np
import tensorflow as tf

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
        self.save_buffer(path)

    def load(self, path, play=None):
        logger.log(f'\nLoading weights from {path}')
        self.model.load_weights(path)
        self.load_buffer(path)

    def save_buffer(self, path):
        self.replay.save(path)

    def load_buffer(self, load_fn, path):
        self.replay.load(load_fn, path)

