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
    try:
        image_pixels = fr.load_image_file(f)
        face_landmarks = fr.face_landmarks(image_pixels)[0]
        mouth = []
        mouth.extend(face_landmarks['top_lip'])
        mouth.extend(face_landmarks['bottom_lip'])
        return np.array(sum(mouth, ()))
    except:
        print(f'{f}:landmarks not detected')
        return []

def perform_pca(lm_vecs):
    pca = PCA(n_components=20)
    mean = np.mean(lm_vecs)
    norm_lm_vec = (lm_vecs - mean) / np.std(lm_vecs)
    return pca.fit_transform(norm_lm_vec)

def upsample(pca):
    
    


def main():
    lock = Lock()
    if len(argv) < 2:
        print('Must have file path argument.')
        exit()
    global file_dir
    file_dir = argv[1]
    if not path.exists('./mouth_features'):
        mkdir('./mouth_features')
    for folder in glob.glob(file_dir+"/*"):
        files = glob.glob(f'{folder}/*.jpg')
        with Pool(initializer=init_child, initargs=(lock,)) as po:
            vals = po.map(extract_mouth_landmarks, files)
            vals = list(filter(any, vals))
            vals = np.array(vals)
            pca = perform_pca(vals)
            upsample(pca)
            exit()
            folder_name = folder.split('/')[-1]
            if not path.exists(f'./mouth_features/{folder_name}'):
                mkdir(f'./mouth_features/{folder_name}')
            np.save(f'./mouth_features/{folder_name}/processed_mouth_shapes', pca)

if __name__ == '__main__':
    main()



