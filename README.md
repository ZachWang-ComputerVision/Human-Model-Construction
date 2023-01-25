# Vertex2Image
 
This project is based on the SMPL and UNet++ model to construct a targeted human figure. Vertex2Image uses SMPL vertices' spatial information to collect color and motion information. Vertex2Image is presenting a human figure model, not constructing an actual human model. With a given camera direction, we can determine which vertices can be seen, then feeding the vertices' information into the model to present what the scene should look like.

Folder structure:
Vertex2Image
    | checkpoint
    | data
        | person_1
            | images_256by256   (ground truth images)
            | color_inputs.json (color information)
            | vert_rot.json     (joint rotation information)
            | camera.pkl        (camera parameters)
        | person_2
            | ......
        | person_n
            | ......
    | output
    | .py files .....

\
To run the model, just run "py train.py". If you want to customize to your dataset, the prerequisit steps are images (extracted from videos) and SMPL information (vertices and joint rotation). It is not neccessary to extract images. One can easily modify the code and run directly with a video source. For using CV2, make sure to convert the image to RBG format before feeding the frames into the model. I may implement the video input method in the future.

The requirements.txt does nto include torch package. Any 1.XX version of Pytorch works with the code. If you do tno want to train on GPU, just set "CUDA = False" in the train.py file.