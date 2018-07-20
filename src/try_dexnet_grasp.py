import os
import itertools
import sys
import glob
import h5py
import numpy as np
import trimesh
import random
import math
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import tensorflow as tf
from tf_grasping.eval import load_configs, latest_checkpoint_path, EvaluationNetwork
from tf_grasping.loss import loss_function_from_config
from tf_grasping.data.dataset_factory import dataset_from_name
from tf_grasping.net.models import model_from_config

sys.path.append('..')

import lib
import lib.utils
from lib.config import (config_mesh_dir, config_output_collected_dir,
                        config_output_processed_dir, config_mesh_dir,
                        project_dir)
from lib import vrep
vrep.simxFinish(-1)
import simulator as SI

# Use the spawn_headless = False / True flag to view with GUI or not
spawn_params = {
    'port':
    19997,
    'ip':
    '127.0.0.1',
    'vrep_path':
    '/opt/V-REP_PRO_EDU_V3_5_0_Linux/vrep.sh',
    'scene_path':
    '/home/agariepy/grasping/grasp-sim/scenes/grasp_scene_big_table.ttt',
    'exit_on_stop':
    True,
    'spawn_headless':
    False,
    'spawn_new_console':
    True
}

query_params = {
    'rgb_near_clip': 0.01,
    'depth_near_clip': 0.01,
    'rgb_far_clip': 10.,
    'depth_far_clip': 1.25,
    'camera_fov': 70 * np.pi / 180,
    'resolution': 300,
    'p_light_off': 0.25,
    'p_light_mag': 0.1,
    'reorient_up': False,
    'randomize_texture': False,
    'randomize_colour': False,
    'randomize_lighting': False,
    'texture_path': os.path.join(project_dir, 'texture.png')
}


def load_mesh(mesh_path):
    """Loads a mesh from file &computes it's centroid using V-REP style."""

    mesh = trimesh.load_mesh(mesh_path)

    # V-REP encodes the object centroid as the literal center of the object,
    # so we need to make sure the points are centered the same way
    center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
    mesh.vertices -= center
    return mesh


ExperimentResult = namedtuple('ExperimentResult', [
    'mesh_path', 'full_image', 'raw_net_output', 'dexnet2_positive',
    'lift_positive'
])


class SimulationExperiment:
    OBJET_DROP_HEIGHT = 0.5
    CAMERA_HEIGHT = 0.7
    CAMERA_FOV = query_params['camera_fov']
    RANDOM_TRANSLATION = 0.3
    GRIPPER_HEIGHT_OFFSET = 0.1
    MODEL_PATH = '/mnt/datasets/dev-tensorboard/supervised-baseline_dexnet2_stn_2018-06-19 16:42:23/'

    def __init__(self, simulator, mesh_list, tf_session):
        self.simulator = simulator
        self.mesh_list = mesh_list
        self.tf_session = tf_session
        if not os.path.exists(config_output_collected_dir):
            os.makedirs(config_output_collected_dir)

        # Load tf model
        configs = load_configs(self.MODEL_PATH)
        batch_size = 1
        model = model_from_config(configs['model'])
        self.loss = loss_function_from_config(batch_size, configs['loss'],
                                              None)
        dataset = dataset_from_name(batch_size, configs['dataset'],
                                    configs['loss'])
        checkpoint_path = latest_checkpoint_path(
            configs['file']['model_directory'])
        self.network = EvaluationNetwork(session, checkpoint_path, dataset,
                                         model, self.loss)

    def try_lift(self):
        pregrasp, postgrasp = self.simulator.run_threaded_candidate()
        if pregrasp is None or postgrasp is None:
            return False
        success = bool(int(postgrasp['all_in_contact']))
        return success

    def image_relative_coord_to_meter(self, x):
        x_angle = x * self.CAMERA_FOV
        return math.tan(x_angle) * self.CAMERA_HEIGHT

    def parse_output(self, output):
        gripper_height = output[0, -1]
        gripper_height *= 0.03597205301927577
        gripper_height += 0.677358905451288

        rectangle = self.loss.parse_network_output(output)[0]
        x, y, angle = rectangle.x, rectangle.y, rectangle.angle
        x = self.image_relative_coord_to_meter(x)
        y = -1 * self.image_relative_coord_to_meter(
            y)  # y is flipped in image space
        angle *= -1  # y is flipped in image space
        is_positive = output[0, 0] < output[0, 1]

        return x, y, angle, gripper_height, is_positive

    def grasp_mesh(self, mesh_path):
        # Load the mesh from file here, so we can generate grasp candidates
        # and access object-specifsc properties like inertia.
        mesh = load_mesh(mesh_path)

        # Compute an initial object resting pose by dropping the object from a
        # given position / height above the workspace table
        mass = mesh.mass_properties['mass'] * 10
        com = mesh.mass_properties['center_mass']
        inertia = mesh.mass_properties['inertia'] * 5
        self.simulator.load_object(mesh_path, com, mass, inertia.flatten())

        initial_pose = self.simulator.get_object_pose()
        random_rotation = lib.utils.format_htmatrix(
            lib.utils.rot_z(random.uniform(0, 2 * math.pi)).dot(
                lib.utils.rot_y(random.uniform(0, 2 * math.pi)).dot(
                    lib.utils.rot_x(random.uniform(0, 2 * math.pi)))))
        initial_pose = random_rotation.dot(initial_pose)
        initial_pose[:3, 3] = [0, 0, self.OBJET_DROP_HEIGHT]

        sim.run_threaded_drop(initial_pose)

        # Reset the object on each grasp attempt to its resting pose. Note this
        # doesn't have to be done, but it avoids instances where the object may
        # subsequently have fallen off the table
        object_pose = sim.get_object_pose()
        object_pose[0, 3] = random.uniform(-self.RANDOM_TRANSLATION,
                                           self.RANDOM_TRANSLATION)
        object_pose[1, 3] = random.uniform(-self.RANDOM_TRANSLATION,
                                           self.RANDOM_TRANSLATION)
        sim.set_object_pose(object_pose)

        gripper_pose = np.dot(
            lib.utils.format_htmatrix(lib.utils.rot_z(np.pi)),
            lib.utils.format_htmatrix(lib.utils.rot_x(np.pi)))
        gripper_pose[2, 3] = self.CAMERA_HEIGHT
        self.simulator.set_gripper_pose(gripper_pose)

        camera_pose = gripper_pose
        images, _ = self.simulator.query(camera_pose, **query_params)
        depth_image = images[0, 3]
        depth_image = np.expand_dims(depth_image, axis=0)
        depth_image = np.expand_dims(depth_image, axis=3)

        raw_out = self.network.feed_forward_batch_raw(depth_image)
        x, y, angle, gripper_height, is_positive = self.parse_output(raw_out)

        gripper_pose = (lib.utils.format_htmatrix(
            lib.utils.rot_z(-np.pi / 2 + angle)).dot(
                lib.utils.format_htmatrix(lib.utils.rot_x(np.pi))))
        gripper_pose[0, 3] = x
        gripper_pose[1, 3] = y
        gripper_pose[
            2,
            3] = self.CAMERA_HEIGHT - gripper_height + self.GRIPPER_HEIGHT_OFFSET
        self.simulator.set_gripper_pose(gripper_pose)

        lift_result = self.try_lift()

        return ExperimentResult(mesh_path, depth_image, raw_out, is_positive,
                                lift_result)

    def run_all_meshed(self):
        results = []
        for m in itertools.cycle(self.mesh_list):
            result = self.grasp_mesh(m)
            print("NET POSITIVE: %s, LIFT WORKED: %s" %
                  (str(result.dexnet2_positive), str(result.lift_positive)))
            results.append(result)
        return results


if __name__ == '__main__':

    meshes = glob.glob(os.path.join(config_mesh_dir, '*'))
    meshes = [
        os.path.join(config_mesh_dir, m) for m in meshes if any(
            x in m for x in ['.stl', '.obj'])
    ]

    sim = SI.SimulatorInterface(**spawn_params)
    with tf.Session() as session:
        experiment = SimulationExperiment(sim, meshes, session)
        experiment.run_all_meshed()
