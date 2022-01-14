import gym
import numpy as np


def apply_wrappers(env):
    env = StateWrapper(env)
    #if env.model_type.startswith('arm'):
    #    return RewardWrapper(env)
    return env


class StateWrapper(gym.Wrapper):

    def reset(self):
        return self._state_extract(super().reset())

    def step(self, action):
        a_eff = self._rescale_action(action)
        next_state, reward, done, info = super().step(a_eff)
        return self._state_extract(next_state), reward, done, info

    def _state_extract(self, state):
        return (((self.data.userdata[3:3+self.model.nu] - 0.75)/(1.05-0.75) -0.5)*2).copy()

    def _rescale_action(self, action):
        return (action.copy() + 1)/2.

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.unwrapped.action_space.shape))

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(self.unwrapped.action_space.shape))

    @property
    def process_state(self):
        return (self.full_state, self.joint_state)

    @property
    def full_state(self):
        return self.unwrapped._get_obs()

    @property
    def joint_state(self):
        return self.data.qpos[:self.unwrapped.nq]


class RewardWrapper(gym.Wrapper):

    def reset(self):
        self.targets = [np.random.uniform(-3,3, size=(3,)) for _ in range(1000)]
        return super().reset()

    def _get_reward_arm(self):
        endeff = self.data.get_site_xpos(self.tracking_str)

        indexes = []
        count = 0
        for idx in range(len(self.targets)):
            if np.linalg.norm(self.targets[idx] - endeff) < 0.1:
                indexes.append(idx)
                count += 1
        for idx in sorted(indexes, reverse=True):
            self.targets.pop(idx)
        return count

    def _get_reward(self, reward):
        if self.model_type == 'arm750':
            return self._get_reward_arm()
        else:
            return reward

    def step(self, a):
        state, reward, done, info = super().step(a)
        return state, self._get_reward(reward), done, info
