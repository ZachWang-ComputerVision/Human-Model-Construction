import glob, os
from PIL import Image
import numpy as np
import h5py
import cv2


""" Apply segmentation and Resize images """
def get_segmentation(dataset_folders):
    for folder in dataset_folders:
        masks = h5py.File(folder + "/masks.hdf5", 'r')['masks']
        
        count = 0
        emp = np.zeros(3)

        caps = cv2.VideoCapture(folder + "/video.mp4")
        next_frame, frame = caps.read()
        
        while next_frame:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = masks[count]
            for i, row in enumerate(mask):
                for j, item in enumerate(row):
                    if item == 0:
                        frame[i, j, :] = emp
            frame = Image.fromarray(frame)
            frame.save(folder + "/seg_images/{0:05d}.png".format(count)) 
            frame = frame.resize((512, 512), resample=Image.ANTIALIAS)
            frame.save(folder + "/images_512by512/{0:05d}.png".format(count))
            
            if count % 100 == 0:
                print("in folder: ", folder, "count: ", count)
            next_frame, frame = caps.read()
            count += 1



""" Resize Images """
def resize_img(dataset_folders):
    for folder in dataset_folders:
        for i, path in enumerate(glob.glob(folder + "/images_512by512/*.png")):
            with Image.open(path, "r") as img:
                # img = np.array(img)
                # img = Image.fromarray(img)
                img = img.resize((256, 256), resample=Image.ANTIALIAS)
                img.save(folder + "/images_256by256/{0:05d}.png".format(i))


""" Rename the files """

def rename_files(paths):
    # paths = "./data/person_8/vertices/*.npy"
    for path in glob.glob(paths):
        folder_path, file_name = path.split("\\")
        file_name = file_name[:-4]
        num = int(file_name)
        os.rename(path, folder_path + f"/{num:05}.npy")