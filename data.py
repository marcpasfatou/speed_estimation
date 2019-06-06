import pandas as pd
from tqdm import tqdm
import os

DATA_PATH = '/home/marc/PycharmProjects/speed_estimation/data/'
TRAIN_PATH = '/home/marc/PycharmProjects/speed_estimation/data/train.mp4'
TEST_PATH = '/home/marc/PycharmProjects/speed_estimation/data/test.mp4'
TRAIN_LABEL_PATH = '/home/marc/PycharmProjects/speed_estimation/data/train.txt'
TRAIN_FOLDER = '/home/marc/PycharmProjects/speed_estimation/data/train'
TEST_FOLDER = '/home/marc/PycharmProjects/speed_estimation/data/test'

train_frames = 20400
test_frames = 10798

def dataset_constructor(vid_path, img_path, tot_frames, dataset_type):
    meta_dict = {}
    tqdm.write('extracting frames..')
    os.system('ffmpeg -i ' + vid_path + ' ' + os.path.join(img_path, '%06d.jpg') + ' -hide_banner')

    tqdm.write('constructing dataset...')
    labels = list(pd.read_csv(TRAIN_LABEL_PATH, header=None, squeeze=True))
    imgs = sorted(list(os.listdir(TRAIN_FOLDER)))
    assert (len(labels) == train_frames)
    for idx, frame in enumerate(imgs):
        img_path = os.path.join(TRAIN_FOLDER, imgs[idx])
        frame_speed = float('NaN') if dataset_type == 'test' else labels[idx]
        meta_dict[idx] = [img_path, idx, frame_speed]
    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed']

    tqdm.write('writing meta to csv')
    meta_df.to_csv(os.path.join(DATA_PATH, dataset_type + '_meta.csv'), index=False)

    print("done dataset_constructor")
    return meta_df



#dataset_constructor(TRAIN_PATH, TRAIN_FOLDER, train_frames, 'train')
#dataset_constructor(TEST_PATH, TEST_FOLDER, test_frames, 'test')