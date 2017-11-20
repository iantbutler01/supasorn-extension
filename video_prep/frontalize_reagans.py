from face_frontalization import *
import cv2 as cv
from glob import glob
from sys import argv, exit
from os import path
from os import mkdir
import numpy as np
import scipy.io as io
from multiprocessing import Pool

MODEL_PATH = '../face_frontalization/face_frontalization/frontalization_models/'
MODEL = threedee.ThreeD_Model(MODEL_PATH + 'model3Ddlib.mat', name='model_dlib')

def forwardify(image_path, model3D, num, folder_name):
    img = cv.imread(image_path, 1)
    landmarks = facial_feature_detector.get_landmarks(img)
    proj_matrix, camera_matrix, rmat, tvec = camera_calibration.estimate_camera(model3D, landmarks[0])
    eyemask = np.asarray(io.loadmat(MODEL_PATH+'eyemask.mat')['eyemask'])
    _, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    cv.imwrite(f'/media/pibrain/extern/frontalized_faces/{folder_name}/{num}.jpg', frontal_sym)

def handle_frames(frame_path):
    folder_name = frame_path.split('/')[-2]
    for j, image_path in enumerate(glob(frame_path+"/*")):
        try:
            forwardify(image_path, MODEL, frame_path.split('/')[-1].split('-')[-1].split('.')[0], folder_name)
        except Exception as e:
            print(e)

def main():
    if len(argv) < 2:
        print('Need to supply path to face images.')
        exit(-1)
    file_dir = argv[-1]
    if not path.exists(f'./frontalized_faces'):
        mkdir(f'/media/pibrain/extern/frontalized_faces')
    check_resources.check_dlib_landmark_weights()

    for sub_folder_path in glob(argv[1]+"/*"):
        folder_name = sub_folder_path.split('/')[-1]
        if not path.exists(f'./frontalized_faces/{folder_name}'):
            mkdir(f'/media/pibrain/extern/frontalized_faces/{folder_name}')
        with Pool() as p:
            p.map(handle_frames, glob(sub_folder_path+"/*"))

if __name__ == '__main__':
    main()

