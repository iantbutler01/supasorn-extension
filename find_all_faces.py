import face_recognition as fr
import glob
import PIL as p
from os import mkdir
from os import path
from multiprocessing import Pool, Lock


def init_child(lock_):
    global lock
    lock = lock_


def find_faces(f):
    new_dir_names = f.split('/')[-3:]
    with lock:
        if not path.exists(f'/Volumes/extern/faces/{new_dir_names[-2]}'):
            mkdir(f'/Volumes/extern/faces/{new_dir_names[-2]}')
        if not path.exists(f'/Volumes/extern/faces/{new_dir_names[-2]}/{new_dir_names[-1]}'):
            mkdir(f'/Volumes/extern/faces/{new_dir_names[-2]}/{new_dir_names[-1]}')
    image_pixels = fr.load_image_file(f)
    face_locations = fr.face_locations(image_pixels)
    image = p.Image.open(f)
    if not face_locations:
        return
    for i,(top, right, bottom, left) in enumerate(face_locations):
        face = image.crop((left, top, right, bottom))
        face.save(f'/Volumes/extern/faces/{new_dir_names[-2]}/{new_dir_names[-1]}/face_{i}.jpg')


def main():
    lock = Lock()

    files = glob.glob("/Volumes/extern/out/**/*.jpg", recursive=True)
    if not path.exists('/Volumes/extern/faces'):
        mkdir('/Volumes/extern/faces')

    with Pool(initializer=init_child, initargs=(lock,)) as p:
        p.map(find_faces, files)


if __name__ == '__main__':
    main()
