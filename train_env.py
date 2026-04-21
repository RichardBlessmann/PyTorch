import time
import numpy as np
import os
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

import udp_sender as udp
import pybullet as p

import build_obstacle_course as obst
from agent import Agent


#initial_xyzs  = spawn point
#initial_rpys  = facing direction
#obstacles     = world layout
#num_drones    = agents
#physics       = realism

#1. create env
#2. env.reset()
#3. spawn obstacles
#4. run loop





start = np.array([0, 0, 1.0])
goal = np.array([5.0, 0.0, 1.0])
start_rpy = [0.0, 0.0, 1.57]

agent = Agent()

# Create environmentW
env = VelocityAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    gui=True,
    obstacles=True,
    initial_xyzs = np.array([start]), #1 meter high
    initial_rpys = np.array([start_rpy]) #roll, pitch, yaw in radian
)
#Time to initialize

time.sleep(1)
# 2. RESET (IMPORTANT!)
obs, info = env.reset()

# =================================================
# add random objects to fly through! later! first just fly through
# =================================================

#obst.spawn_curved_tower_course(start, goal, client=env.CLIENT, difficulty=1)

#visual markers /start/end
p.loadURDF("sphere2.urdf", start, globalScaling=0.3)
p.loadURDF("sphere2.urdf", goal, globalScaling=0.3)

if os.path.isfile('drone_model.pth'):
    agent.load('drone_model.pth')
    print("Netz wurde geladen")
else :
    print('wak')


print("Simulation started")

EPISODES = 500
STEPS_PER_EPISODE = 2400

for episode in range(EPISODES):

    # =================================================
    # Reset environment each episode
    # =================================================
    obs_raw, info = env.reset()

    # get initial state
    state = env._getDroneStateVector(0)
    pos = state[0:3]
    vel = state[10:13]

    obs = np.concatenate([pos, goal, vel])

    prev_dist = np.linalg.norm(pos - goal)
    episode_reward = 0

    for step in range(STEPS_PER_EPISODE):

        # =================================================
        # Ask agent for action
        # =================================================
        action, value = agent.act(obs)

        vx, vy, vz, yaw_rate = action

        action_env = np.array([
            [vx * 2.0, vy * 2.0, vz * 1.0, yaw_rate]
        ])

        # =================================================
        # Step simulation
        # =================================================
        _, _, terminated, truncated, info = env.step(action_env)

        # =================================================
        # Read new drone state
        # =================================================
        state = env._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]

        next_obs = np.concatenate([pos, goal, vel])

        # =================================================
        # Reward shaping
        # =================================================
        dist = np.linalg.norm(pos - goal)

        #closer to goal -> higher reward!
        #farther away -> negative reward
        reward = (prev_dist - dist) * 10.0
        prev_dist = dist

        # reached goal bonus
        if dist < 0.25:
            reward += 100
            terminated = True

        done = terminated or truncated

        episode_reward += reward

        # =================================================
        # Store experience
        # =================================================
        agent.store(obs, action, reward, value, done)

        obs = next_obs

        # optional render speed
        time.sleep(1 / 240)

        if done:
            break

    # =================================================
    # Learn after episode
    # =================================================
    agent.update()

    print(f"Episode {episode} | Reward: {episode_reward:.2f} | Final Dist: {dist:.2f}")

agent.save("drone_model.pth")
env.close()
print("Finished")