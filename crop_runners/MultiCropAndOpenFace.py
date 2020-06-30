"""
.. module MultiCropAndOpenFace
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
# import crop_runners.CropAndOpenFace as CropAndOpenFace
import CropAndOpenFace
from multiprocessing import Pool

def make_vids(input_path, output_path, emotions = False):
    """
    Return list of vids not processed yet given a path.
    NEW: Also return only those that have emotions file
    :param path: Path to video directory
    :type path: str
    :return: list of vids to do
    """
    folder_components = set(os.path.join(output_path, x) for x in os.listdir(output_path))

    # this is to find all .avi videos that are in the given input dir, even recursively
    # might not always be needed (somewhat time consuming)
    paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".avi"):
                paths.append(os.path.join(root, file))

    # this is for processing only those videos we have emotion annotations for, for classifier training purposes
    to_process = []
    if emotions:
        for p in paths:
            x = os.path.splitext(os.path.split(p)[1])[0]
            if x + '_emotions.csv' in os.listdir('/home/emil/emotion_annotations'):
                if (os.path.join(output_path, x) + '_cropped' not in folder_components
                        or 'hdfs' not in os.listdir(
                            os.path.join(output_path,
                                         x + '_cropped'))):
                    to_process.append(p)
    else:
        for p in paths:
            x = os.path.splitext(os.path.split(p)[1])[0]
            if (os.path.join(output_path, x) + '_cropped' not in folder_components
                    or 'hdfs' not in os.listdir(
                        os.path.join(output_path,
                                     x + '_cropped'))):
                    to_process.append(p)

    # this is optional, but I'm not using the videos from 11PM-7AM anyways, so I won't process those
    # get time for each vid
    patient = os.path.basename(to_process[0]).split('_')[0]
    time_path = os.path.join('/nas/ecog_project/derived/processed_ecog/',
                             patient, 'full_day_ecog/vid_start_end_merge.csv')
    times = pd.read_csv(time_path)
    to_process_time_filtered = []
    print("I'm about to filter everything that's not from 7AM-23PM. You sure you want that?")
    for p in to_process:
        curr_time = times[times['filename']==os.path.basename(p)]
        if len(curr_time) == 0:  # quickfix for bug on vid start time file creation
            continue
        if  23  > curr_time['hour'].iloc[0] > 5:  # add everything that is between 7-23
            to_process_time_filtered.append(p)
        elif 23 == curr_time['hour'].iloc[0] and curr_time['minute'].iloc[0] <= 5:  # edge cases
            to_process_time_filtered.append(p)
        elif curr_time['hour'].iloc[0] == 6 and curr_time['minute'].iloc[0] >= 55:
            to_process_time_filtered.append(p)

    return to_process_time_filtered

# # These are just a collection of patient_day_vid key and ACTUAL crop file path as value, so a lookup dictionary
# def make_crop_and_nose_files(path):
#     crop_file = os.path.join(path, 'crop_files_list.txt')
#     nose_file = os.path.join(path, 'nose_files_list.txt')
#
#     if not os.path.exists(crop_file):
#         crop_path = sys.argv[sys.argv.index('-c') + 1]
#         crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
#         json.dump(crop_txt_files, open(crop_file, mode='w'))
#
#     if not os.path.exists(nose_file):
#         nose_path = sys.argv[sys.argv.index('-n') + 1]
#         nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)
#         json.dump(nose_txt_files, open(nose_file, mode='w'))
#
#     return json.load(open(crop_file)), json.load(open(nose_file))


def crop(vid):
    im_dir = os.path.join(output_path,os.path.splitext(vid)[0].split('/')[-1] + '_cropped')
    if not os.path.exists(im_dir):
        os.mkdir(im_dir)
    # for idx, vid in tqdm(enumerate(vids)):  # this sequentially bc ffmpeg is an absolute pain in the ass wrt I/O load
    try:
        CropAndOpenFace.duration(vid)
        # CropAndOpenFace.VideoImageCropper(
        #     vid=vid,
        #     im_dir=im_dir,
        #     crop_txt_files=crop_txt_files,
        #     nose_txt_files=nose_txt_files)
        CropAndOpenFace.crop_and_resize(vid, im_dir)
    except CropAndOpenFace.DurationException as e:
        print(str(e) + '\t' + vid)



if __name__ == '__main__':
    input_path = sys.argv[sys.argv.index('-id') + 1]
    output_path = sys.argv[sys.argv.index('-od') + 1]

    vids = make_vids(input_path,output_path)
    # crop_txt_files, nose_txt_files = make_crop_and_nose_files(output_path)
    os.chdir(output_path)
    # into chunks
    chunks = []
    chunk_size = 20
    for i in range(0, len(vids), chunk_size):
        chunks.append(vids[i:i + chunk_size])
    # pool = Pool(8)
    # pool.map(crop_and_openface,vids)

    for chunk in tqdm(chunks):  # we don't have enough memory. hence do the following in chunks
        with Pool(8) as p:
            test = list(tqdm(p.imap(crop, chunk), total=len(chunk)))  # create the frames for openface via multiproc
        im_dirs = [os.path.join(output_path, os.path.splitext(vid)[0].split('/')[-1] + '_cropped') for vid in chunk]
        CropAndOpenFace.run_open_face(im_dirs)  # run openface on it

    sys.exit(0)
