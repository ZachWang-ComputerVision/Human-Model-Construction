import json
import pickle
import numpy as np
from PIL import Image
import glob
import h5py
import json
from projection_utils import get_corresponding_grid

resolution = (256, 256)


dataset_folders = []
for path in glob.glob("./data/*"):
    dataset_folders.append(path)
dataset_folders = dataset_folders[0:4]
print(dataset_folders)


def get_files_path(path):
    file_path = []
    for file in glob.glob(path):
        file_path.append(file)
    return file_path


def read_image(path):
    with Image.open(path, "r") as img:
        data = np.array(img)
    return data


# def read_vertex(path):
#     data = np.load(path, allow_pickle=True)
#     return data

def read_vertex(path):
    with open(path, "r") as jf:
        data = json.loads(jf.read())
    return data



""" Collect vertex colors from images """
def collect_vertex_colors():
    # this is universal, so I just need to load once
    with open("./data/person_1/faces.json", "r") as jf:
        faces = json.loads(jf.read())['faces']

    # this is universal, so I just need to load once   
    with open("./data/person_1/camera.pkl", 'rb') as pf:
        cam = pickle._Unpickler(pf)
        cam.encoding = 'latin1'
        cam = cam.load()

    proj_mat = np.array([[cam['camera_f'][0], 0., cam['camera_c'][0]],
                        [0., cam['camera_f'][1], cam['camera_c'][1]], 
                        [0.,                 0.,                 1.]])
    view_vector = np.array([[0.], [0.], [1.]])

    # The following function is to convert homogeneous matrix to view vector
    # def convert_to_view_vec(R_mat):
    #   R = np.transpose(R_mat)
    #   view_vec = np.dot(R, np.array([[0], [0], [1]]))
    #   return view_vec


    for folder in dataset_folders:
        img_paths = get_files_path(folder + "/images_256by256/*.png")
        vertices = read_vertex(folder + "/smpl.json")
        # ver_paths = get_files_path(folder + "/vertices/*.npy")

        vertex_colors = []
        for _ in range(6890):
            vertex_colors.append([])

        vertex_maps = []

        for i in range(len(list(vertices.keys()))):
        # for i in range(len(ver_paths)):
            img = read_image(img_paths[i])
            if vertices.has_key(f"{i}"):
                v_xyz = vertices[f"{i}"]["frame_verts"]
                v_xyz = np.array(v_xyz)
                # v_xyz = read_vertex(ver_paths[i])

                vertex_map = get_corresponding_grid(img, v_xyz, faces, view_vector, proj_mat, resolution)
                vertex_maps.append(vertex_map)

                # smpl_img = np.zeros(resolution)
                for r, row in enumerate(vertex_map):
                    for c, item in enumerate(row):
                        if np.isnan(item) == False:
                            rgb = img[r, c]
                            vertex_colors[int(item)].append(rgb)
                            # smpl_img[r][c] = 255
                
                # smpl_img = smpl_img.astype(np.uint8)
                # smpl_img = Image.fromarray(smpl_img).convert("L")
                # smpl_img.save(folder + f"/smpl_img/frame{i}.png")

        for j in range(6890):
            if len(vertex_colors[j]) != 0:
                color = np.array(vertex_colors[j])
                vertex_colors[j] = np.mean(color, axis=0)
        
        print("finished collecting vertex color in folder")

        # Store inputs into a JSON file
        color_inputs = []
        for m, vertex_map in enumerate(vertex_maps):
            input_img = np.zeros((resolution[0], resolution[1], 3))

            for vr, row in enumerate(vertex_map):
                for vc, item in enumerate(row):
                    if np.isnan(item) == False:
                        input_img[vr,vc] = vertex_colors[int(item)]
                    # else:
                    #     vertex_map[vr,vc] = -1.
            input_img = np.array([input_img[:,:,0], input_img[:,:,1], input_img[:,:,2]])
            color_inputs.append(input_img.tolist())
            vertex_maps[m] = vertex_map.tolist()

        with open(folder + "/color_inputs.json", "w") as wif:
            json.dump(color_inputs, wif)
        
        with open(folder + "/index_inputs.json", "w") as wvf:
            json.dump(vertex_maps, wvf)
        
        print("finished dumping inputs to folder: ", folder)




""" Get Rotation Angles """

def get_vertex_rotation(folder="./data/person_5/"):
    with open(folder + "vertex_label.json", "r") as jf:
        seg = json.loads(jf.read())
    keys = list(seg.keys())

    with open(folder + "index_inputs.json", "r") as vi:
        indices = json.loads(vi.read())

    vert_label = [0 for _ in range(6890)]
    for i, key in enumerate(keys):
        vertex = seg[key]
        for v in vertex:
            vert_label[v] = i

    seg_corresponding_joint = [21,2,16,4,10,7,6,9,13,14,8,15,17,22,5,23,18,19,12,11,3,1,20,0]

    poses = h5py.File(folder + "reconstructed_poses.hdf5", 'r')
    poses = np.array(poses["pose"])
    poses = np.reshape(poses, (-1, 24, 3))
    print("poses shape: ", poses.shape)
    import sys
    rot_inputs = np.zeros((poses.shape[0], 3, 256, 256))
    index = np.array(indices)
    for f, frame in enumerate(indices):
        for r, row in enumerate(frame):
            for c, item in enumerate(row):
                if np.isnan(item) == False:
                    label = vert_label[int(item)]
                    seg_label = seg_corresponding_joint.index(label)
                    rot = np.reshape(poses[f][seg_label], (1, 3))
                    rot_inputs[f, :, r, c] = rot

    rot_inputs = rot_inputs.tolist()
    with open(folder + "vert_rot.json", "w") as drf:
        json.dump(rot_inputs, drf)