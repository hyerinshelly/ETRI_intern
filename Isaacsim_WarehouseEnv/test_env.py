import numpy as np

from omni.isaac.synthetic_utils import SyntheticDataHelper
from pxr import UsdGeom, Gf, Sdf, Usd, PhysxSchema, PhysicsSchema, PhysicsSchemaTools, Semantics

from str import STR
from warehouse_environment import Environment

from gym import spaces


def line_seg_closest_point(v0, v1, p0):
    # Project p0 onto (v0, v1) line, then clamp to line segment
    d = v1 - v0
    q = p0 - v0

    t = np.dot(q, d) / np.dot(d, d)
    t = np.clip(t, 0, 1)

    return v0 + t * d


def line_seg_distance(v0, v1, p0):
    p = line_seg_closest_point(v0, v1, p0)

    return np.linalg.norm(p0 - p)


def is_going_forward(prev_pose, curr_pose):
    prev_pose = 0.01 * prev_pose
    curr_pose = 0.01 * curr_pose

    bottom_left_corner = np.array([0, 0])
    top_left_corner = np.array([0, 10.668])
    top_right_corner = np.array([6.711, 10.668])
    bottom_right_corner = np.array([6.711, 0])

    d0 = line_seg_distance(bottom_left_corner, top_left_corner, curr_pose)
    d1 = line_seg_distance(top_left_corner, top_right_corner, curr_pose)
    d2 = line_seg_distance(top_right_corner, bottom_right_corner, curr_pose)
    d3 = line_seg_distance(bottom_right_corner, bottom_left_corner, curr_pose)

    min_d = np.min([d0, d1, d2, d3])

    which_side = np.array([0, 0])
    if min_d == d0:
        which_side = top_left_corner - bottom_left_corner
    elif min_d == d1:
        which_side = top_right_corner - top_left_corner
    elif min_d == d2:
        which_side = bottom_right_corner - top_right_corner
    elif min_d == d3:
        which_side = bottom_left_corner - bottom_right_corner

    which_size_unit = which_side / np.linalg.norm(which_side)

    curr_vel = curr_pose - prev_pose
    curr_vel_norm = np.linalg.norm(curr_vel)

    curr_vel_unit = np.array([0, 0])
    # checking divide by zero
    if curr_vel_norm:
        curr_vel_unit = curr_vel / curr_vel_norm

    return np.dot(curr_vel_unit, which_size_unit)

class TestEnv:
    metadata = {"render.modes": ["human"]}

    def __init__(self, omni_kit, z_height=0, max_resets=10, updates_per_step=3, steps_per_rollout=500):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(224, 224, 6), dtype=np.uint8)
        self.noise = 0.05

        # every time we update the stage, this is how much time will be simulated
        self.dt = 1 / 30.0
        self.omniverse_kit = omni_kit
        self.sd_helper = SyntheticDataHelper() # get groundtruth data, sensor data
        self.warehouse = Environment(self.omniverse_kit) ##################################

        # make environment z up
        self.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)

        # generate warehouse
        self.shape = [6, 6]
        self.warehouse.generate_warehouse(self.shape)
        self.warehouse.generate_lights()

        # spawn robot
        self.STR = STR(self.omniverse_kit)
        self.initial_loc = [-1800,5000,0]
        self.STR.spawn(Gf.Vec3d(self.initial_loc[0], self.initial_loc[1], 5), 0)
        self.prev_pose = [-1800,5000,0]
        self.current_pose = [-1800,5000,0]

        # switch kit camera to jetracer camera
        self.STR.activate_camera()

        # start simulation
        self.omniverse_kit.play()

        # Step simulation so that objects fall to rest
        # wait until all materials are loaded
        frame = 0
        print("simulating physics...")
        while frame < 60 or self.omniverse_kit.is_loading():
            self.omniverse_kit.update(self.dt)
            frame = frame + 1
        print("done after frame: ", frame)

        self.initialized = False
        self.numsteps = 0
        self.numresets = 0
        self.maxresets = 10

    def calculate_reward(self):
        # Current and last positions
        pose = np.array([self.current_pose[0], self.current_pose[1]])
        prev_pose = np.array([self.prev_pose[0], self.prev_pose[1]])

        # Finite difference velocity calculation
        vel = pose - prev_pose
        vel_norm = vel
        vel_magnitude = np.linalg.norm(vel)
        if vel_magnitude > 0.0:
            vel_norm = vel / vel_magnitude

        # Distance from the center of the track
        dist = min(abs([-1800,6000]-pose))
        self.dist = dist

        racing_forward = is_going_forward(prev_pose, pose)
        reward = racing_forward * self.current_speed * np.exp(-dist ** 2 / 0.05 ** 2)

        return reward

    def is_dead(self):
        done = False

        # kill the episode after 500 steps
        if self.numsteps > 500:
            done = True

        return done

    def step(self, action):
        print("Number of steps ", self.numsteps)

        # print("Action ", action)

        self.STR.command(action*300)
        frame = 0
        total_reward = 0
        reward = 0
        while frame < 3:
            self.omniverse_kit.update(self.dt)
            obs = self.STR.observations()
            self.prev_pose = self.current_pose
            self.current_pose = obs["pose"]
            self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
            self.current_forward_velocity = obs["local_linear_velocity"][0]

            reward = self.calculate_reward()
            done = self.is_dead()

            total_reward += reward
            frame = frame + 1

        gt = self.sd_helper.get_groundtruth(["rgb", "camera"])

        currentState = gt["rgb"][:, :, :3]

        if not self.initialized:
            self.previousState = currentState

        img = np.concatenate((currentState, self.previousState), axis=2)

        self.previousState = currentState

        self.numsteps += 1
        if done:
            print("robot is dead")

        if self.numsteps > 500:
            done = True
            print("robot stepped 500 times")

        return img, reward, done, {}

    def reset(self):
        if self.numresets % self.maxresets == 0:
            self.warehouse.reset(self.shape)

        if not self.initialized:
            state, reward, done, info, = self.step([0, 0])
            self.initialized = True

        # every time we reset, we move the robot to a random location, and pointing along the direction of the road
        loc = [-1800,5000,0]
        # the random angle offset can be increased here
        rot = [0,0,0]
        self.STR.teleport(Gf.Vec3d(loc[0], loc[1], 5), rot, settle=True)

        obs = self.STR.observations()
        self.current_pose = obs["pose"]
        self.current_speed = np.linalg.norm(np.array(obs["linear_velocity"]))
        self.current_forward_velocity = obs["local_linear_velocity"][0]

        # waiting for loading
        if self.numresets % self.maxresets == 0:
            frame = 0
            while self.omniverse_kit.is_loading():  # or frame < 750:
                self.omniverse_kit.update(self.dt)
                frame += 1

        gt = self.sd_helper.get_groundtruth(["rgb", "camera"])
        currentState = gt["rgb"][:, :, :3]

        img = np.concatenate((currentState, currentState), axis=2)

        self.numsteps = 0
        self.previousState = currentState
        self.numresets += 1

        return img






