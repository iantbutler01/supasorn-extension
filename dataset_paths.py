from glob import glob


get_folder_name = lambda x: x.split('/')[-1]
audio_folders = list(map(get_folder_name, glob('./features/mfccs/*')))
frame_folders = list(map(get_folder_name, glob('./features/mouth_features/*')))
data_set_names = list(set(audio_folders) & set(frame_folders))
audio_features = list(map(lambda x: f'/home/pibrain/Desktop/fis/features/mfccs/{x}', data_set_names))
mouth_features = list(map(lambda x: f'/home/pibrain/Desktop/fis/features/mouth_features/{x}', data_set_names))
print(audio_features)
print(mouth_features)
