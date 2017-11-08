import face_recognition as fr
import glob
import PIL as p
from os import mkdir
from os import path
from sys import argv
from sys import exit
from multiprocessing import Pool, Lock
import numpy as np
from sklearn.decomposition import PCA

def init_child(lock_):
    global lock
    lock = lock_


def extract_mouth_landmarks(f):
    image_pixels = fr.load_image_file(f)
    face_landmarks = fr.face_landmarks(image_pixels)[0]
    mouth = []
    mouth.extend(face_landmarks['top_lip'])
    mouth.extend(face_landmarks['bottom_lip'])
    return np.array(sum(mouth, ()))

def perform_pca(lm_vecs):
    pca = PCA()
    mean = np.mean(lm_vecs)
    norm_lm_vec = (lm_vecs - mean) / np.std(lm_vecs)
    pca.fit(norm_lm_vec)
    coef = pca.components_
    print(coef)
    return coef

def main():
    lock = Lock()
    if len(argv) < 2:
        print('Must have file path argument.')
        exit()
    global file_dir
    file_dir = argv[1]
    files = glob.glob(f'{file_dir}/out/**/*.jpg', recursive=True)
    with Pool(initializer=init_child, initargs=(lock,)) as po:
        vals = po.map(extract_mouth_landmarks, files)
    vals = np.array(vals)
    pca = perform_pca(vals)


if __name__ == '__main__':
    main()



