import face_recognition as fr
import glob
import PIL as p
from os import mkdir
from os import path
from sys import argv
from sys import exit
from multiprocessing import Pool, Lock


def init_child(lock_):
    global lock
    lock = lock_


def find_faces(f):
    image_pixels = fr.load_image_file(f)
    face_locations = fr.face_locations(image_pixels)
    image = p.Image.open(f)
    if not face_locations:
        return
    new_dir_names = f.split('/')[-3:]
    with lock:
        if not path.exists(f'{file_dir}/faces/{new_dir_names[-2]}'):
            mkdir(f'{file_dir}/faces/{new_dir_names[-2]}')
        if not path.exists(f'{file_dir}/faces/{new_dir_names[-2]}/{new_dir_names[-1]}'):
            mkdir(f'{file_dir}/faces/{new_dir_names[-2]}/{new_dir_names[-1]}')
    for i,(top, right, bottom, left) in enumerate(face_locations):
        face = image.crop((left, top, right, bottom))
        face.save(f'{file_dir}/faces/{new_dir_names[-2]}/{new_dir_names[-1]}/face_{i}.jpg')


def main():
    lock = Lock()
    if len(argv) < 2:
        print('Must have file path argument.')
        exit()
    global file_dir
    file_dir = argv[1]
    files = glob.glob(f'{file_dir}/out/**/*.jpg', recursive=True)
    if not path.exists(f'{file_dir}/faces'):
        mkdir(f'{file_dir}/faces')

    with Pool(initializer=init_child, initargs=(lock,)) as p:
        p.map(find_faces, files)


if __name__ == '__main__':
    main()
