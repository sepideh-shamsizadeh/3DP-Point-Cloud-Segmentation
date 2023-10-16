import numpy as np
import random
import math
import time
import struct
import os

# pyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# a nice training progress bar
from tqdm import tqdm

# visualization
import open3d as o3d
import plotly.graph_objects as go



numpoints = 20000 # [number of points]
max_dist = 15     # [meters]
min_dist = 4      # [meters]

# transform distances to squares (code optimization)
max_dist *= max_dist
min_dist *= min_dist

size_float = 4
size_small_int = 2

dataset_path =  "dataset"


    
semantic_kitti_color_scheme = {
0 : [0, 0, 0],        # "unlabeled"
1 : [0, 0, 255],      # "outlier"
10: [245, 150, 100],  # "car"
11: [245, 230, 100],  # "bicycle"
13: [250, 80, 100],   # "bus"
15: [150, 60, 30],    # "motorcycle"
16: [255, 0, 0],      # "on-rails"
18: [180, 30, 80],    # "truck"
20: [255, 0, 0],      # "other-vehicle"
30: [30, 30, 255],    # "person"
31: [200, 40, 255],   # "bicyclist"
32: [90, 30, 150],    # "motorcyclist"
40: [255, 0, 255],    # "road"
44: [255, 150, 255],  # "parking"
48: [75, 0, 75],      # "sidewalk"
49: [75, 0, 175],     # "other-ground"
50: [0, 200, 255],    # "building"
51: [50, 120, 255],   # "fence"
52: [0, 150, 255],    # "other-structure"
60: [170, 255, 150],  # "lane-marking"
70: [0, 175, 0],      # "vegetation"
71: [0, 60, 135],     # "trunk"
72: [80, 240, 150],   # "terrain"
80: [150, 240, 255],  # "pole"
81: [0, 0, 255],      # "traffic-sign"
99: [255, 255, 50],   # "other-object"
252: [245, 150, 100], # "moving-car"
253: [200, 40, 255],  # "moving-bicyclist"
254: [30, 30, 255],   # "moving-person"
255: [90, 30, 150],   # "moving-motorcyclist"
256: [255, 0, 0],     # "moving-on-rails"
257: [250, 80, 100],  # "moving-bus"
258: [180, 30, 80],   # "moving-truck"
259: [255, 0, 0],     # "moving-other-vehicle"
}


label_remap = {
0 :  0, # "unlabeled"
1 :  0, # "outlier"
10:  2, # "car"
11:  2, # "bicycle"
13:  2, # "bus"
15:  2, # "motorcycle"
16:  2, # "on-rails"
18:  2, # "truck"
20:  2, # "other-vehicle"
30:  2, # "person"
31:  2, # "bicyclist"
32:  2, # "motorcyclist"
40:  1, # "road"
44:  1, # "parking"
48:  1, # "sidewalk"
49:  1, # "other-ground"
50:  2, # "building"
51:  2, # "fence"
52:  2, # "other-structure"
60:  1, # "lane-marking"
70:  2, # "vegetation"
71:  2, # "trunk"
72:  2, # "terrain"
80:  2, # "pole"
81:  2, # "traffic-sign"
99:  2, # "other-object"
252: 2, # "moving-car"
253: 2, # "moving-bicyclist"
254: 2, # "moving-person"
255: 2, # "moving-motorcyclist"
256: 2, # "moving-on-rails"
257: 2, # "moving-bus"
258: 2, # "moving-truck"
259: 2, # "moving-other-vehicle"
    }

remap_color_scheme = [
  [0, 0, 0],
  [0, 255, 0],
  [0, 0, 255]
]


def sample(pointcloud, labels, numpoints_to_sample):
    """
        INPUT
            pointcloud          : list of 3D points
            labels              : list of integer labels
            numpoints_to_sample : number of points to sample
    """
    tensor = np.concatenate((pointcloud, np.reshape(labels, (labels.shape[0], 1))), axis= 1)
    tensor = np.asarray(random.choices(tensor, weights=None, cum_weights=None, k=numpoints_to_sample))
    pointcloud_ = tensor[:, 0:3]
    labels_ = tensor[:, 3]
    labels_ = np.array(labels_, dtype=np.int_)
    return pointcloud_, labels_


def readpc(pcpath, labelpath, reduced_labels=True):
    """
    INPUT
        pcpath         : path to the point cloud ".bin" file
        labelpath      : path to the labels ".label" file
        reduced_labels : flag to select which label encoding to return
                        [True]  -> values in range [0, 1, 2]   -- default
                        [False] -> all Semantic-Kitti dataset original labels
    """
    
    pointcloud, labels = [], []

    with open(pcpath, "rb") as pc_file, open(labelpath, "rb") as label_file:
        byte = pc_file.read(size_float*4)
        label_byte = label_file.read(size_small_int)
        _ = label_file.read(size_small_int)

        while byte:
            x,y,z, _ = struct.unpack("ffff", byte)      # unpack 4 float values
            label = struct.unpack("H", label_byte)[0]   # unpach 1 Unsigned Short value
            
            d = x*x + y*y + z*z       # Euclidean norm

            if min_dist<d<max_dist:
                pointcloud.append([x, y, z])
                if reduced_labels:            # for reduced labels range
                    labels.append(label_remap[label])
                else:                         # for full labels range
                    labels.append(label)
            
            byte = pc_file.read(size_float*4)
            label_byte = label_file.read(size_small_int)
            _ = label_file.read(size_small_int)
        

    pointcloud  = np.array(pointcloud)
    labels      = np.array(labels)

    # return fixed_sized lists of points/labels (fixed size: numpoints)
    return sample(pointcloud, labels, numpoints)


def remap_to_bgr(integer_labels, color_scheme):
  bgr_labels = []
  for n in integer_labels:
    bgr_labels.append(color_scheme[int(n)][::-1])
  np_bgr_labels = np.array(bgr_labels)
  return np_bgr_labels

def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            )
        )
    )
    fig.show()


def visualize3DPointCloud(np_pointcloud, np_labels):
  """
  INPUT
      np_pointcloud : numpy array of 3D points
      np_labels     : numpy array of integer labels
  """
  assert(len(np_pointcloud) == len(np_labels))


  pcd = o3d.geometry.PointCloud()
  v3d = o3d.utility.Vector3dVector

  # set geometry point cloud points
  pcd.points = v3d(np_pointcloud)

  # scale color values in range [0:1]
  pcd.colors = o3d.utility.Vector3dVector(np_labels / 255.0)

  # replace rendering function
  o3d.visualization.draw_geometries = draw_geometries

  # visualize the colored point cloud
  o3d.visualization.draw_geometries([pcd])

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)
    

def default_transforms():
    return transforms.Compose([
                                Normalize(),
                                ToTensor()
                              ])

class PointCloudData(Dataset):
    def __init__(self, dataset_path, transform=default_transforms(), start=0, end=1000):
        """
          INPUT
              dataset_path: path to the dataset folder
              transform   : transform function to apply to point cloud
              start       : index of the first file that belongs to dataset
              end         : index of the first file that do not belong to dataset
        """
        self.dataset_path = dataset_path
        self.transforms = transform

        self.pc_path = os.path.join(self.dataset_path, "sequences", "00", "velodyne")
        self.lb_path = os.path.join(self.dataset_path, "sequences", "00", "labels")

        self.pc_paths = os.listdir(self.pc_path)
        self.lb_paths = os.listdir(self.lb_path)
        assert(len(self.pc_paths) == len(self.lb_paths))

        self.start = start
        self.end   = end

        # clip paths according to the start and end ranges provided in input
        self.pc_paths = self.pc_paths[start: end]
        self.lb_paths = self.lb_paths[start: end]

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
      item_name = str(idx + self.start).zfill(6)
      pcpath = os.path.join(self.pc_path, item_name + ".bin")
      lbpath = os.path.join(self.lb_path, item_name + ".label")

      # load points and labels
      pointcloud, labels = readpc(pcpath, lbpath)

      # transform
      torch_pointcloud  = torch.from_numpy(pointcloud)
      torch_labels      = torch.from_numpy(labels)

      return torch_pointcloud, torch_labels


if __name__ == '__main__':
    # define point cloud example index and absolute paths
    pointcloud_index = 146
    pcpath    = os.path.join(dataset_path, "sequences", "00", "velodyne", str(pointcloud_index).zfill(6) + ".bin"  )
    labelpath = os.path.join(dataset_path, "sequences", "00", "labels",   str(pointcloud_index).zfill(6) + ".label")

    # load pointcloud and labels with original Semantic-Kitti labels
    pointcloud, labels = readpc(pcpath, labelpath, False)
    labels = remap_to_bgr(labels, semantic_kitti_color_scheme)
    print("Semantic-Kitti original color scheme")
    visualize3DPointCloud(pointcloud, labels)

    # load pointcloud and labels with remapped labels
    pointcloud, labels = readpc(pcpath, labelpath)
    labels = remap_to_bgr(labels, remap_color_scheme)
    print("Remapped color scheme")
    visualize3DPointCloud(pointcloud, labels)