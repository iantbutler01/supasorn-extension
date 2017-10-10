import face_recognition as fr
from glob import glob
from sys import argv,exit
import os


def is_reagan(folder):
    known_image = fr.load_image_file("known_reagan.jpg")
    reagan_encoding = fr.face_encodings(known_image)[0]
    for sub_folder in glob(f'{folder}/*'):
        rm_folder(sub_folder)
        for img in glob(f'{sub_folder}/*.jpg'):
            try:
                unknown_image = fr.load_image_file(img)
                unknown_encoding = fr.face_encodings(unknown_image)[0]
                results = fr.compare_faces([reagan_encoding], unknown_encoding)
                if not results[0]:
                    os.remove(img)
                    rm_folder(sub_folder)
            except Exception as e:
                print(e)
                os.remove(img)
                rm_folder(sub_folder)
                continue


def rm_folder(folder_path):
    print(folder_path)
    try:
        if not os.listdir(folder_path):
            os.rmdir(folder_path)
    except Exception as e:
        print(e)


def main():
    if len(argv) < 2:
        print('Need path to potential Reagans.')
        exit(-1)
    global file_dir
    file_dir = argv[1]

    for folder in glob(f'{file_dir}/*'):
        is_reagan(folder)



if __name__ == '__main__':
    main()


