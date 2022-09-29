import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import torchvision.transforms
import torch
import numpy as np
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.data.unlabeled_dataset import UnlabeledKeypointsDataset
import sys
import pyzed.sl as sl
import cv2
import wandb
from pathlib import Path
from skimage import io
import torchvision

# Adapted from: https://github.com/stereolabs/zed-opencv/blob/master/python/zed-opencv.py

help_string = "[s] Save side by side image, [q] Quit"
path = "./"

count_save = 0

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    

def process_key_event(zed, key) :
    global count_save

    if key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")


def get_wandb_model():
    # checkpoint_reference = "airo-box-manipulation/clothes/model-2gerrzs4:v3"
    # checkpoint_reference = 'airo-box-manipulation/iros2022_0/model-1ip92rpu:v2'
    checkpoint_reference = 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'

    # download checkpoint locally (if not already cached)
    run = wandb.init(project="clothes", entity="airo-box-manipulation")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    model = KeypointDetector.load_from_checkpoint(Path(artifact_dir) / "model.ckpt",backbone_type='ConvNeXtUnet')
    return model



def main() :

    model = get_wandb_model()

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    print(image_size.width, image_size.height)

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)

    transform  = torchvision.transforms.Resize((256,256))

    key = ' '
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            width = image_size.width
            start = int(width - 540) // 2
            end = start + 540
            image_BGR = image_ocv[:, start:end]
            cv2.imshow("Image", image_BGR)

            image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB).astype(np.float32)
            image_RGB /= 255.0

            with torch.no_grad():
                print(image_RGB.shape)
                image_batched = image_RGB[np.newaxis, :, :, :]
                image_batched_torch = torch.Tensor(image_batched)
                image_batched_permuted = image_batched_torch.permute((0, 3, 1, 2))
                print(image_batched_permuted.shape)
                batch = transform(image_batched_permuted)
                print(batch.shape)
                channel = 0
                output = model(batch)[:,channel]

                overlayed = overlay_image_with_heatmap(batch, output)
                first = overlayed[0]
                first = first.permute((1,2,0))
                first = first.numpy()
                first = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
                first = cv2.resize(first, (1024, 1024))

            cv2.imshow("Output", first)


            key = cv2.waitKey(10)
            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()