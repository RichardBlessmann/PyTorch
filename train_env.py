import time
import numpy as np
import os
import json

from dateutil.tz import EPOCH

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
goal = np.array([0.5, 0.0, 1.0])
start_rpy = [0.0, 0.0, 1.57]

agent = Agent()

# Create environmentW
env = VelocityAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    gui=False,
    obstacles=True,
    initial_xyzs = np.array([start]), #1 meter high
    initial_rpys = np.array([start_rpy]), #roll, pitch, yaw in radian
    pyb_freq=480,
    ctrl_freq=240,
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
log_data = []
world_scale = 5.0


def exp_goal_reward(pos, goal, scale=1.0):
    dist = np.linalg.norm(pos - goal)
    reward = np.exp(-scale * dist)
    return reward

def _computeReward(self):
    state = self._getDroneStateVector(0)

    pos = state[0:3]
    att = state[7:10]
    vel = state[10:13]
    ang_vel = state[13:16]

    targer_pos = np.array([0, 0, 1])
    pos_err = np.linalg.norm(goal - pos)

    att_err = np.linalg.norm(att)
    vel_err = np.linalg.norm(vel)
    ang_vel_err = np.linalg.norm(ang_vel)

    W_pos_err = 1
    W__att = 0
    W_vel = 0
    W_ang_vel = 0

    reward = (-1 * W_pos_err * pos_err) + (-1 * W__att * att) + (-1 * W_vel * vel) + (-1 * W_ang_vel * ang_vel)

    if pos_err < 0.0001:
        reward += 1

    return reward


for episode in range(EPISODES):

    # =================================================
    # Reset environment each episode
    # =================================================
    obs_raw, info = env.reset()

    # get initial state
    state = env._getDroneStateVector(0)
    pos = state[0:3]
    vel = state[10:13]


    direction = goal - pos
    dist = np.linalg.norm(direction)

    obs = np.concatenate([
        direction / 5.0,
        vel / 3.0,
        [dist / 5.0]
    ])

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

        direction = goal - pos
        dist = np.linalg.norm(direction)

        next_obs = np.concatenate([
            direction / world_scale,
            vel / 3.0,
            [dist / world_scale]
        ])

        # =================================================
        # Reward shaping
        # =================================================
        dist = np.linalg.norm(pos - goal)
        #closer to goal -> higher reward!
        #farther away -> negative reward

        progress = (prev_dist - dist)

        # 1. MAIN SIGNAL (movement)
        reward = progress

        # 2. SMALL SHAPING (helps near goal)
        #reward += 0.01 * np.exp(-6.0 * dist)

        # 3. TIME PENALTY (prevents doing nothing)
        #reward -= 0.002

        done = False
        # 4. GOAL BONUSES (graduated)
        if dist < 0.2:
            reward += 1.0

        if dist < 0.1:
            reward += 3.0
            done = True

        # 5. CRASH PENALTY
        if pos[2] < 0.1:
            reward -= 5.0
            done = True

        done = terminated or truncated

        episode_reward += reward

        # =================================================
        # Store experience
        # =================================================
        agent.store(obs, action, reward, value, done)

        obs = next_obs
        if episode % 100 == 0 and episode > 0:
            log_data.append({
                "episode": episode,
                "step": step,
                "pos": pos.tolist(),
                "vel": vel.tolist(),
                "goal": goal.tolist()
            })
        # optional render speed
        #time.sleep(1 / 240)
        if done:
            break


    # =================================================
    # Learn after episode
    # =================================================

    if episode % 100 == 0 and episode > 0:
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/drone_log_ep_{episode}.json"

        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Saved log: {filename}")

        log_data = []  # reset nach speichern

    agent.update()

    print(f"Episode {episode} | Reward: {episode_reward:.2f} | Final Dist: {dist:.2f}")

agent.save("drone_model.pth")
env.close()
print("Finished")


