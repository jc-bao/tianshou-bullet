import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

controller_config = load_controller_config(default_controller='OSC_POSE')
env =  GymWrapper(suite.make(
    env_name='TwoArmHandover', # try with other tasks like "Stack" and "Door"
    robots=["Panda"]*2,  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=False,
    use_object_obs=True,
    use_camera_obs=False,
    reward_shaping=True,
    controller_configs=controller_config,
    control_freq = 20
))

# reset the environment
obs = env.reset()
print(env.action_space.shape, env.observation_space.shape)

for i in range(1000):
    print(obs)
    action = np.random.randn(env.action_dim) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    # env.render()  # render on display