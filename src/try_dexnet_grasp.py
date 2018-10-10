import os
import itertools
import sys
import glob
import h5py
import numpy as np
import random
import math
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataset import G3DB, KIT

import tensorflow as tf
from tf_grasping.img_utils import to_rgb_img
from tf_grasping.eval import load_configs, latest_checkpoint_path
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


class Simulator:
    OBJET_DROP_HEIGHT = 0.5
    CAMERA_HEIGHT = 0.7
    CAMERA_FOV = query_params['camera_fov']
    RANDOM_TRANSLATION = 0.2
    GRIPPER_HEIGHT_OFFSET = 0.04
    MODEL_PATH = '/mnt/datasets/tensorboard-stn/yolo-like-loss_dexnet2_stn_2018-09-06 16:18:21/'

    def __init__(self, vrep, tf_session):
        self.vrep = vrep
        self.tf_session = tf_session
        if not os.path.exists(config_output_collected_dir):
            os.makedirs(config_output_collected_dir)

        # Load tf model
        configs = load_configs(self.MODEL_PATH)
        self._init_model(configs)
        self.loss = loss_function_from_config(1, configs['loss'], None)

    @property
    def _camera_pose(self):
        pose = np.dot(
            lib.utils.format_htmatrix(lib.utils.rot_z(np.pi)),
            lib.utils.format_htmatrix(lib.utils.rot_x(np.pi)))
        pose[2, 3] = self.CAMERA_HEIGHT
        return pose

    def _init_model(self, configs):
        self.img_placeholder = tf.placeholder(
            tf.float32, shape=[1, 300, 300, 1])
        self.model = model_from_config(configs['model'])
        self.net_out = self.model(
            self.img_placeholder, output_size=9, is_training=False)

        self.visualization = self.model.visualize_transforms_batch()

        checkpoint_path = latest_checkpoint_path(
            configs['file']['model_directory'])
        self._load_checkpoint(checkpoint_path)

    def feed_forward_image(self, img):
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)

        return self.tf_session.run([self.net_out, self.visualization],
                                   feed_dict={self.img_placeholder: img})

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

        return x, y, angle, rectangle, gripper_height, is_positive

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

    def _grasp_to_pose(self, x, y, angle, height):
        gripper_pose = (lib.utils.format_htmatrix(lib.utils.rot_z(angle)).dot(
            lib.utils.format_htmatrix(lib.utils.rot_x(np.pi))))
        gripper_pose[0, 3] = x
        gripper_pose[1, 3] = y
        gripper_pose[2, 3] = height
        return gripper_pose

    def show_visualization(self, depth_image, rectangle, visualization):
        display_image = to_rgb_img(depth_image)
        display_image = rectangle.draw_on_image(display_image)
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(display_image)
        axes[0, 1].imshow(visualization[1][0, :, :, 0])
        axes[1, 0].imshow(visualization[2][0, :, :, 0])
        axes[1, 1].imshow(visualization[3][0, :, :, 0]* 0.04414705 + 0.69519629)
        plt.show()

    def find_best_grasp_pose(self, depth_image):
        depth_image = depth_image[50:350, 50:350]  # Central crop
        raw_out, visualization = self.feed_forward_image(depth_image)
        # gripper_height = np.min(visualization[-1]) * 0.04414705 + 0.69519629
        # gripper_height = self.CAMERA_HEIGHT - gripper_height + self.GRIPPER_HEIGHT_OFFSET
        # gripper_height = 0.08
        x, y, angle, rectangle, gripper_height, is_positive = self.parse_output(raw_out)
        gripper_height = max(0.055, self.CAMERA_HEIGHT - gripper_height + self.GRIPPER_HEIGHT_OFFSET)
        print(raw_out)
        self.show_visualization(depth_image, rectangle, visualization)
        return self._grasp_to_pose(x, y, angle, gripper_height)

    def grasp_mesh(self, mesh_file, mass, com, inertia):
        self.drop_object(mesh_file, mass, com, inertia)
        self.random_object_pose()
        self.reset_gripper_pose()

        depth_image = self.capture_depth_image()
        gripper_pose = self.find_best_grasp_pose(depth_image)
        self.move_gripper(gripper_pose)
        lift_result = self.try_lift()
        return lift_result

    def run_all_meshed(self, dataset):
        results = []
        random_indices = np.arange(len(dataset))
        np.random.shuffle(random_indices)
        for i in random_indices:
            is_positive = self.grasp_mesh(*dataset[i])
            print('LIFT WORKED: %s' % str(is_positive))
            results.append(is_positive)
        return results


def load_simulator(session=None):
    if session is None:
        session = tf.InteractiveSession()
    vrep = SI.SimulatorInterface(**spawn_params)
    return Simulator(vrep, session)


if __name__ == '__main__':
    dataset = KIT(KIT_PATH)
    with tf.Session() as session:
        simulator = load_simulator(session)
        simulator.run_all_meshed(dataset)
