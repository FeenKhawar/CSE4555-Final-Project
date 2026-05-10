'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
smpl_webuser = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, project_root)
sys.path.insert(0, smpl_webuser)

import numpy as np
import pyrender
import trimesh
from smpl.smpl_webuser.serialization import load_model
import cv2

## Load SMPL model (here we load the female model)
model_path = os.path.join(os.path.dirname(__file__), '../../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
m = load_model(os.path.abspath(model_path))

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * 0.2 #  np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose[0] = 0 #np.pi

## Get vertices and faces from SMPL model
vertices = np.array(m.r)  # m.r triggers the chumpy computation
faces = np.array(m.f)

## Build a trimesh and then a pyrender mesh
tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

## Create pyrender scene
scene = pyrender.Scene(ambient_light=np.array([0.8, 0.8, 0.8, 1.0]), bg_color=np.array([0.0, 0.0, 0.0, 1.0]))
scene.add(mesh)

## Camera — place it in front of the mesh looking at origin
w, h = 640, 480
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=w/h)
camera_pose = np.array([
    [1, 0, 0, 1.4],
    [0, 1, 0, 0.8],   # slight y offset to center the body
    [0, 0, 1, 3.5],   # z=2.5 in front
    [0, 0, 0, 1  ],
])
scene.add(camera, pose=camera_pose)

## Directional light from above-front
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
light_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 2],
    [0, 0, 1, 2],
    [0, 0, 0, 1],
])
scene.add(light, pose=light_pose)

## Multiple lights for better visibility
for light_pos in [[-1, -1, 1], [1, -1, 1], [0, 1, 1]]:
    light = pyrender.PointLight(color=np.ones(3), intensity=3.0)
    lp = np.eye(4)
    lp[:3, 3] = light_pos
    scene.add(light, pose=lp)

# print("vertices min/max:", vertices.min(axis=0), vertices.max(axis=0))

pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(w, h))

# ## Render
# r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
# color, _ = r.render(scene)
# r.delete()

# ## Show with OpenCV (pyrender returns RGB, cv2 needs BGR)
# cv2.imshow('render_SMPL', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
# print('Press any key while on the display window')
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()