import numpy as np
import json
from PIL import Image
import glob


def get_folder_paths(folder="./data/"):
    dataset_folders = []
    for path in glob.glob(folder + "*"):
        dataset_folders.append(path)
    dataset_folders = dataset_folders[0:5]
    print("dataset paths: ", dataset_folders)
    return dataset_folders


def get_files_path(path):
    file_path = []
    for file in glob.glob(path):
        file_path.append(file)
    return file_path


def read_image(path):
    with Image.open(path, "r") as img:
        data = np.array(img)
    return data


def read_vertex(path):
    with open(path, "r") as jf:
        data = json.loads(jf.read())
    return data


def read_img_generator(paths):
  for file_path in glob.glob(paths):
    with Image.open(file_path) as img:
      img = np.array(img)
      img = np.array([img[:,:,0], img[:,:,1], img[:,:,2]])
      yield img


def read_input_map_generator(path):
  with open(path, "r") as jf:
    data = json.loads(jf.read())
  for input in data:
    yield input


""" Generate Dot Images To Check Projection Vadility """
def generate_dot_img():
    resolution = (256, 256)

    with open("./data/person_1/index_inputs.json", "r") as rft:
        data = json.loads(rft.read())

    data = np.array(data)
    for v, v_map in enumerate(data):
        smpl_img = np.zeros(resolution)
        for r, row in enumerate(v_map):
            for c, item in enumerate(row):
                if np.isnan(item) == False:
                # if item != -1.:
                    smpl_img[r][c] = 255.
        
        smpl_img = smpl_img.astype(np.uint8)
        smpl_img = Image.fromarray(smpl_img).convert("L")
        smpl_img.save("./data/person_1/smpl_img/frame{i}.png")