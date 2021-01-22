# import necessary libraries
import numpy as np
import gym
from gym import spaces

class Betting(gym.Env):

    # actions available
    UP = 0
    DOWN = 1

    def __init__(self, data):
        super(Betting, self).__init__() # gym.Env의 __init__ 호출

        # data 정의
        self.data = data
        self.size = len(data)  # size of the data

        # randomly assign the inital location of agent
        self.agent_position = np.random.randint(self.size - 1)

        # respective actions of agents : up, down
        self.action_space = spaces.Discrete(2)

        # set the observation space to (1,) to represent agent position
        self.observation_space = spaces.Box(low=0, high=self.size, shape=(1,), dtype=np.uint8)

    def step(self, action):
        info = {}  # additional information

        reward = 0

        # UP, DOWN 맞으면 reward=1, 틀리면 맞을 때까지 반복
        if action == self.UP:
            if self.data[self.agent_position] < self.data[self.agent_position + 1]:
                reward += 1
                self.agent_position += 1
            else:
                reward += 0
        elif action == self.DOWN:
            if self.data[self.agent_position] > self.data[self.agent_position + 1]:
                reward += 1
                self.agent_position += 1
            else:
                reward += 0
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # 더 이상 데이터가 없을 경우, done
        done = bool(self.agent_position == self.size - 1)

        return np.array([self.agent_position]).astype(np.uint8), reward, done, info

    def render(self, mode='console'):
        '''
            render the state
        '''
#         if mode != 'console':
#             raise NotImplementedError()

#         for pos in range(self.size):
#             if pos == self.agent_position:
#                 print("X", end='')
#             else:
#                 print('.', end='')
#             print('')

    def reset(self):
        # -1 to ensure agent inital position will not be at the end state
        self.agent_position = np.random.randint(self.size - 2)

        return np.array([self.agent_position]).astype(np.uint8)

    def close(self):
        pass