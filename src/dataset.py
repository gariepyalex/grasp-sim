import numpy as np
import os
import glob
import trimesh
import tempfile

import sys
sys.path.append('..')
import lib.utils


class Dataset:
    def __init__(self, dataset_path):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, i):
        raise NotImplementedError()

    def to_mesh_tempfile(self, mesh):
        mesh_file = tempfile.NamedTemporaryFile(mode='w', suffix='.obj')
        trimesh.io.export.export_mesh(mesh, mesh_file, file_type='obj')
        return mesh_file


class G3DB(Dataset):
    def __init__(self, dataset_path):
        """
        args:
          - dataset_path: path of the cloned github repo https://github.com/kleinash/G3DB
        """
        self._path = os.path.join(dataset_path, 'objects', 'grasp_meshes',
                                 'final_mesh')
        self._object_weights = np.loadtxt(
            os.path.join(self.path, 'mesh_weights.csv'), delimiter=',')
        self._object_paths = self._load_object_paths()

    def __len__(self):
        return len(self._object_paths)

    def _load_object(self, i):
        mesh_path = self._object_paths[i]
        mesh = trimesh.load_mesh(mesh_path)

        # V-REP encodes the object centroid as the literal center of the object,
        # so we need to make sure the points are centered the same way
        center = lib.utils.calc_mesh_centroid(mesh, center_type='vrep')
        mesh.vertices -= center
        mesh.vertices /= 100  # G3DB meshes are in cm, convert to meter

        mesh_file = self.to_mesh_tempfile(mesh)
        return mesh, mesh_file

    def __getitem__(self, i):
        mesh, mesh_file = self._load_object(i)
        mass = self._object_weights[i]
        com = mesh.mass_properties['center_mass']
        inertia = np.eye(3) * 1e-3
        return mesh_file, mass, com, inertia

    def _load_object_paths(self):
        def obj_number(obj_path):
            filename = os.path.basename(obj_path)
            return int(filename.split('_')[0])

        paths = glob.glob(os.path.join(self._path, '*.obj'))
        paths = sorted(paths, key=obj_number)
        return paths


class KIT(Dataset):
    def __init__(self, dataset_path):
        self._object_paths = sorted(glob.glob(os.path.join(self._path, '*.obj')))

    def __len__(self):
        return len(self._object_paths)

    def __getitem__(self, i):
        mesh = trimesh.load_mesh(mesh_path)
        mesh.vertices -= mesh.center_mass
        mesh_file = self.to_mesh_tempfile(mesh)

        mass = 1.0
        center_mass = mesh.center_mass
        inertia = np.eye(3) * 1e-3
        return mesh_file, mass, center_mass, inertia


if __name__ == '__main__':
    path = '/mnt/datasets/grasping/G3DB'
    g3db = G3DB(path)
