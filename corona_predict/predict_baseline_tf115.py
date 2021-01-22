# Same code with UPDOWNprediction.ipynb

# This is to test if custom enviroment created properly
# If the environment don't follow the gym interface, an error will be thrown

from updown_predict_env import Betting
from stable_baselines.common.env_checker import check_env

data = [0,9,7,4,3,5]

env = Betting(data)
check_env(env, warn=True)

# try draw the grid world
obs = env.reset()
env.render()

# import various RL algorithms
from stable_baselines import DQN, PPO2, A2C, ACKTR

# Train the agent
model = ACKTR('MlpPolicy', env, verbose=1).learn(10000)

# running the simulation with trained model to verify result

obs = env.reset()
n_steps = 50
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break

# print the trained policy map
# recall UP = 0, DOWN = 1
for i in range(len(data)):
    obs = [i]
    action, _ = model.predict(obs, deterministic=True)

    if i % len(data) == 0 and i != 0:
        print('')  # newline
    print(action, end="")