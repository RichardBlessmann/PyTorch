import time
import numpy as np
import os
import json

from dateutil.tz import EPOCH

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

#import udp_sender as udp
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
goal = np.array([2.0, 0.0, 1.0])
start_rpy = [0.0, 0.0, 1.57]

agent = Agent()

# Create environmentW
env = VelocityAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    gui=True,
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

EPISODES = 1000
STEPS_PER_EPISODE = 2400
log_data = []
world_scale = 5.0

def compute_distances(pos, goal):
    xy = np.linalg.norm(pos[:2] - goal[:2])
    z = abs(pos[2] - goal[2])
    return xy, z

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

    #targer_pos = goal
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

def _rewardFunc(self, prev_pos, pos):
    xy_dist,z_dist = compute_distances(pos, goal)
    xy_prev_dist, z_prev_dist = compute_distances(prev_pos, goal)
    #xy_dist = np.linalg.norm(pos[:2] - goal[:2])
    #z_dist = abs(pos[2] - goal[2])

    # closer to goal -> higher reward!
    # further away -> negative reward

    goal_vec = goal - pos
    goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)

    progress_xy = xy_prev_dist - xy_dist

    reward = progress_xy

    reward -= 0.5 * z_dist

    reward -= 0.005

    # ---------------------------------------
    #   drone movement vector
    # ---------------------------------------

    velocity = pos - prev_pos
    alignment = np.dot(velocity, goal_dir)
    reward += 0.1 * alignment



    success = dist < 0.1

    speed = np.linalg.norm(velocity)

    if xy_dist < 0.3:
        reward -= 0.05 * speed

    if success:
        reward += 1.0
        done = True

    done = terminated or truncated or success



    return reward, done, pos


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
    # split distance vector new try
    prev_xy_dist = np.linalg.norm(pos[:2] - goal[:2])
    prev_z_dist = np.linalg.norm(pos[2] - goal[2])
    prev_pos = pos.copy()

    episode_reward = 0

    for step in range(STEPS_PER_EPISODE):

        # =================================================
        # Ask agent for action
        # =================================================
        action, value = agent.act(obs)

        # =================================================
        # Step simulation
        # =================================================
        action_env = np.array([action])

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
        reward, done, prev_pos = _rewardFunc(env, prev_pos, pos)

        # =================================================
        # Store experience
        # =================================================
        agent.store(obs, action, reward, value, done)

        obs = next_obs
        #if episode % 100 == 0 and episode > 0:
        #    log_data.append({
        #        "episode": episode,
        #        "step": step,
        #        "pos": pos.tolist(),
        #        "vel": vel.tolist(),
        #        "goal": goal.tolist()
        #    })


        # optional render speed
        #time.sleep(1 / 240)
        if done:
            break


    # =================================================
    # Learn after episode
    # =================================================

    agent.update()

    #if episode % 100 == 0 and episode > 0:
    #    os.makedirs("logs", exist_ok=True)
    #    filename = f"logs/drone_log_ep_{episode}.json"

    #    with open(filename, "w") as f:
    #        json.dump(log_data, f, indent=2)

    #    print(f"Saved log: {filename}")

    #    log_data = []  # reset nach speichern


    print(f"Episode {episode} | Reward: {episode_reward:.2f} | Final Dist: {dist:.2f}")

agent.save("drone_model.pth")
env.close()
print("Finished")


