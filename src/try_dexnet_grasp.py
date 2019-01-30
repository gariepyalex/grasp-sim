import argparse
import os
import itertools
import uuid
import sys
import glob
import h5py
import numpy as np
import random
import math
import pickle
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataset import G3DB, KIT

import tensorflow as tf
from tf_grasping.img_utils import to_rgb_img
from tf_grasping.inference import load_best_network

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
    '/home/agariepy/grasping/grasp-sim/scenes/grasp_scene_robotiq.ttt',
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
    'resolution': 400,
    'p_light_off': 0.25,
    'p_light_mag': 0.1,
    'reorient_up': False,
    'randomize_texture': False,
    'randomize_colour': False,
    'randomize_lighting': False,
    'texture_path': os.path.join(project_dir, 'texture.png')
}

G3DB_PATH = '/mnt/datasets/grasping/G3DB'
KIT_PATH = '/mnt/datasets/grasping/KIT'

GraspAttempt = namedtuple('GraspAttempt', [
    'object_pose', 'grasp_pose', 'image', 'success', 'raw_net_output',
    'mesh_name'
])


class Simulator:
    OBJET_DROP_HEIGHT = 0.5
    CAMERA_HEIGHT = 0.7
    CAMERA_FOV = query_params['camera_fov']
    RANDOM_TRANSLATION = 0.1
    GRIPPER_HEIGHT_OFFSET = 0.04
    MODEL_PATH = '/mnt/datasets/tensorboard-stn/trying-old-schedule_dexnet2_stn_2019-01-27 12:23:33/'

    def __init__(self, vrep, tf_session):
        self.vrep = vrep
        self.tf_session = tf_session
        if not os.path.exists(config_output_collected_dir):
            os.makedirs(config_output_collected_dir)

        # Load tf model
        self._init_model(self.MODEL_PATH)

    @property
    def _camera_pose(self):
        pose = np.dot(
            lib.utils.format_htmatrix(lib.utils.rot_z(np.pi)),
            lib.utils.format_htmatrix(lib.utils.rot_x(np.pi)))
        pose[2, 3] = self.CAMERA_HEIGHT
        return pose

    def _init_model(self, path):
        img_shape = [1, 224, 224, 1]
        self.inference = load_best_network(self.tf_session, path, img_shape)

    def feed_forward_image(self, img):
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)
        return self.inference.feed_forward_batch(img)[0]

    def _load_checkpoint(self, checkpoint_path):
        saver = tf.train.Saver()
        saver.restore(self.tf_session, checkpoint_path)

    def try_lift(self):
        pregrasp, postgrasp = self.vrep.run_threaded_candidate()
        if pregrasp is None or postgrasp is None:
            return False
        success = bool(int(postgrasp['all_in_contact']))
        return success

    def image_relative_coord_to_meter(self, x):
        x *= 3 / 4
        x_angle = x * self.CAMERA_FOV
        return math.tan(x_angle) * self.CAMERA_HEIGHT

    def move_gripper(self, pose_matrix):
        self.vrep.set_gripper_pose(pose_matrix)

    def drop_object(self, mesh_file, mass, com, inertia):
        self.vrep.load_object(mesh_file.name, com, mass, inertia.flatten())
        initial_pose = self.vrep.get_object_pose()
        random_rotation = lib.utils.format_htmatrix(
            lib.utils.rot_z(random.uniform(0, 2 * math.pi)).dot(
                lib.utils.rot_y(random.uniform(0, 2 * math.pi)).dot(
                    lib.utils.rot_x(random.uniform(0, 2 * math.pi)))))
        initial_pose = random_rotation.dot(initial_pose)
        initial_pose[:3, 3] = [0, 0, self.OBJET_DROP_HEIGHT]

        self.vrep.run_threaded_drop(initial_pose)

    @property
    def object_pose(self):
        return self.vrep.get_object_pose()

    @object_pose.setter
    def object_pose(self, pose):
        self.vrep.set_object_pose(pose)

    def random_object_pose(self):
        object_pose = self.vrep.get_object_pose()
        object_pose[0, 3] = random.uniform(-self.RANDOM_TRANSLATION,
                                           self.RANDOM_TRANSLATION)
        object_pose[1, 3] = random.uniform(-self.RANDOM_TRANSLATION,
                                           self.RANDOM_TRANSLATION)
        self.vrep.set_object_pose(object_pose)

    def reset_gripper_pose(self):
        self.vrep.set_gripper_pose(self._camera_pose)

    def capture_depth_image(self):
        images, _ = self.vrep.query(self._camera_pose, **query_params)
        depth_image = images[0, 3]
        return depth_image

    def _grasp_to_pose(self, rectangle, height):
        x, y, angle = rectangle.x, rectangle.y, rectangle.angle
        x = self.image_relative_coord_to_meter(x)
        y = -1 * self.image_relative_coord_to_meter(
            y)  # y is flipped in image space
        angle *= -1  # y is flipped in image space
        gripper_pose = (lib.utils.format_htmatrix(lib.utils.rot_z(angle)).dot(
            lib.utils.format_htmatrix(lib.utils.rot_x(np.pi))))
        gripper_pose[0, 3] = x
        gripper_pose[1, 3] = y
        gripper_pose[2, 3] = height
        return gripper_pose

    def show_visualization(self, depth_image, rectangle):
        display_image = to_rgb_img(depth_image)
        display_image = rectangle.draw_on_image(display_image)
        plt.imshow(display_image)
        plt.show()

    def find_best_grasp_pose(self, depth_image, debug):
        depth_image = depth_image[88:312, 88:312]  # Central crop
        out = self.feed_forward_image(depth_image)
        raw_out = out['raw']
        rectangle = out['rectangle']
        gripper_height = out['gripper_height']

        # gripper_height = max(
        #     0.055,
        #     self.CAMERA_HEIGHT - gripper_height + self.GRIPPER_HEIGHT_OFFSET)
        gripper_height = 0.065
        if debug:
            print(raw_out)
            self.show_visualization(depth_image, rectangle)
        return self._grasp_to_pose(rectangle, gripper_height), raw_out

    def grasp_mesh(self, mesh, debug=True):
        self.drop_object(mesh.file, mesh.mass, mesh.center_mass, mesh.inertia)
        self.random_object_pose()
        self.reset_gripper_pose()

        object_pose = self.object_pose
        depth_image = self.capture_depth_image()
        gripper_pose, raw_out = self.find_best_grasp_pose(depth_image, debug)
        self.move_gripper(gripper_pose)
        success = self.try_lift()
        attempt = GraspAttempt(
            grasp_pose=gripper_pose,
            object_pose=object_pose,
            image=depth_image,
            raw_net_output=raw_out,
            success=success,
            mesh_name=mesh.name)
        return attempt

    def save_attempt(self, output_dir, attempt):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        filename = '%s_%s.pkl' % (attempt.mesh_name, str(uuid.uuid1()))
        print(os.path.join(output_dir, filename))
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(attempt, f)

    def run_all_meshed(self, dataset, output_dir, debug, shuffle=False):
        n_success = 0
        indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            attempt = self.grasp_mesh(dataset[i], debug)
            if attempt.success:
                n_success += 1
            if debug:
                print('LIFT WORKED: %s' % str(attempt.success))
            if output_dir:
                self.save_attempt(output_dir, attempt)
        return n_success / len(dataset)


def load_simulator(session=None):
    if session is None:
        session = tf.InteractiveSession()
    vrep = SI.SimulatorInterface(**spawn_params)
    return Simulator(vrep, session)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    dataset = KIT(KIT_PATH)
    with tf.Session() as session:
        simulator = load_simulator(session)
        accuracy = simulator.run_all_meshed(
            dataset, args.output_dir, debug=args.debug)
        print('Accuracy: ' + str(round(accuracy, 4) * 100) + '%')
