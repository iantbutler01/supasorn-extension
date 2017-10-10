from face_frontalization import *
import cv2 as cv
from glob import glob
from sys import argv, exit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

def forwardify(image_path, model3D):
    img = cv.imread(image_path, 1)
    landmarks = facial_feature_detector.get_landmarks(img)
    proj_matrix, camera_matrix, rmat, tvec = camera_calibration.estimate_camera(model3D, landmarks[0])
    eyemask = np.asarray(io.loadmat('./frontalization_models/eyemask.mat')['eyemask'])
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    plt.figure()
    plt.title('Frontalized no symmetry')
    plt.imshow(frontal_raw[:, :, ::-1])
    plt.figure()
    plt.title('Frontalized with soft symmetry')
    plt.imshow(frontal_sym[:, :, ::-1])
    plt.show()

def main():
    if len(argv) < 2:
        print('Need to supply path to face images.')
        exit(-1)
    check_resources.check_dlib_landmark_weights()
    model = threedee.ThreeD_Model('./frontalization_models/model3Ddlib.mat', name='model_dlib')
    for sub_folder_path in glob(argv[1]+"/*"):
        for image_path in glob(sub_folder_path+"/*"):
            forwardify(image_path, model)

if __name__ == '__main__':
    main()

