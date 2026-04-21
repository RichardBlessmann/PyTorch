import pybullet as p
import random
import numpy as np

import json
import udp_sender as udp

# =========================================================
# TOWER (FIXED: thicker, proper visual blocks)
# =========================================================
def spawn_tower(client, pos, height, width):

    p.loadURDF(
        "cube.urdf",
        basePosition=[pos[0], pos[1], height / 2],
        globalScaling=width,
        physicsClientId=client,
        useFixedBase=True
    )


def get_params(difficulty: float):

    # clamp
    d = np.clip(difficulty, 0.0, 1.0)

    return {
        "num_towers": int(10 + d * 60),          # 10 → 70
        "safe_radius": 2.5 - d * 1.8,            # wide → tight
        "jitter": 0.5 + d * 2.5,                 # smooth → chaotic
        "tower_width": 0.3 + d * 0.6,            # thin → thick
        "tower_height": (2 + d * 10),            # short → tall
    }
# =========================================================
# MAIN COURSE GENERATOR
# =========================================================
def spawn_curved_tower_course(
    start,
    goal,
    client,
    difficulty=0.5
):

    params = get_params(difficulty)

    start = np.array(start)
    goal = np.array(goal)

    path = generate_waypoints(
        start,
        goal,
        n=5,
        jitter=params["jitter"]
    )

    # =====================================================
    # CREATE SCENE DATA HERE (✔ CORRECT PLACE)
    # =====================================================
    scene_data = {
        "start": start.tolist(),
        "goal": goal.tolist(),
        "towers": [],
        "path": path.tolist()
    }

    # -----------------------------
    # draw path
    # -----------------------------
    for i in range(len(path) - 1):
        p.addUserDebugLine(
            path[i],
            path[i + 1],
            [0, 1, 0],
            2.0,
            physicsClientId=client
        )

    # -----------------------------
    # spawn obstacles
    # -----------------------------
    for _ in range(params["num_towers"]):

        for _ in range(20):

            t = random.random()
            pos = start + t * (goal - start)

            direction = goal - start
            perp = np.array([-direction[1], direction[0], 0])
            perp /= (np.linalg.norm(perp) + 1e-8)

            pos += random.uniform(-3, 3) * perp
            pos[2] = random.uniform(0.5, 4)

            if distance_to_path(pos, path) < params["safe_radius"]:
                continue

            tower_height = random.uniform(2, params["tower_height"])

            spawn_tower(
                client=client,
                pos=pos,
                height=tower_height,
                width=params["tower_width"]
            )

            # =================================================
            # ADD TO EXPORT (IMPORTANT PART YOU MISSED)
            # =================================================
            scene_data["towers"].append({
                "pos": pos.tolist(),
                "height": float(tower_height),
                "width": float(params["tower_width"])
            })

            break
    udp.send_data(scene_data)



# =========================================================
# WAYPOINT GENERATION
# =========================================================
def generate_waypoints(start, goal, n=4, jitter=2.0):

    start = np.array(start)
    goal = np.array(goal)

    points = [start]

    for i in range(1, n):
        t = i / n
        base = start + t * (goal - start)

        direction = goal - start
        perp = np.array([-direction[1], direction[0], 0])
        perp = perp / (np.linalg.norm(perp) + 1e-8)

        offset = random.uniform(-jitter, jitter)
        height = random.uniform(-0.5, 0.5)

        point = base + offset * perp
        point[2] = start[2] + height

        points.append(point)

    points.append(goal)
    return np.array(points)


# =========================================================
# DISTANCE TO PATH (UNCHANGED)
# =========================================================
def distance_to_path(point, path):

    p2 = point[:2]

    min_dist = float("inf")

    for i in range(len(path) - 1):

        a = path[i][:2]
        b = path[i + 1][:2]

        ab = b - a
        ap = p2 - a

        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8), 0, 1)
        closest = a + t * ab

        dist = np.linalg.norm(p2 - closest)
        min_dist = min(min_dist, dist)

    return min_dist