"""
.. module MultiCropAndOpenFace
    :synopsis: Script to apply cropping and OpenFace to all videos in a directory.

"""

import glob
import json
import os
import subprocess
from warnings import warn
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import CropAndOpenFace
from timeit import default_timer as timer


def make_vids(input_path, output_path, emotions = False):
    """
    Return list of vids not processed yet given a path.
    NEW: Also return only those that have emotions file
    :param path: Path to video directory
    :type path: str
    :return: list of vids to do
    """
    folder_components = set(os.path.join(output_path, x) for x in os.listdir(output_path))

    #this is to find all .avi videos that are in the given input dir, even recursively
    #might not always be needed (somewhat time consuming)
    paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".avi"):
                paths.append(os.path.join(root, file))

    #this is for processing only those videos we have emotion annotations for
    if emotions:
        to_process = []
        for p in paths:
            x = os.path.splitext(os.path.split(p)[1])[0]
            if x + '_emotions.csv' in os.listdir('/home/emil/emotion_annotations'):
                if (os.path.join(output_path, x) + '_cropped' not in folder_components
                        or 'hdfs' not in os.listdir(
                            os.path.join(output_path,
                                         x + '_cropped'))):
                    to_process.append(p)


    else:
        pat_sess_vid = [os.path.splitext(os.path.split(x)[1])[0] for x in paths]
        to_process = [
            os.path.join(input_path, x + '.avi') for x in pat_sess_vid

            if (os.path.join(output_path, x) + '_cropped' not in folder_components
                or 'hdfs' not in os.listdir(
                        os.path.join(output_path,
                                     os.path.splitext(x)[0] + '_cropped')))
        ]
    return to_process



def make_crop_and_nose_files(path): #FOUND OUT: These are just supposed to be a collection of patient_day_vid key and ACTUAL crop file path as value, so basically a lookup dictionary.
    crop_file = os.path.join(path, 'crop_files_list.txt')
    nose_file = os.path.join(path, 'nose_files_list.txt')

    if not os.path.exists(crop_file):
        crop_path = sys.argv[sys.argv.index('-c') + 1]
        crop_txt_files = CropAndOpenFace.find_txt_files(crop_path)
        json.dump(crop_txt_files, open(crop_file, mode='w'))

    if not os.path.exists(nose_file):
        nose_path = sys.argv[sys.argv.index('-n') + 1]
        nose_txt_files = CropAndOpenFace.find_txt_files(nose_path)
        json.dump(nose_txt_files, open(nose_file, mode='w'))

    return json.load(open(crop_file)), json.load(open(nose_file))


if __name__ == '__main__':
    input_path = sys.argv[sys.argv.index('-id') + 1]
    output_path = sys.argv[sys.argv.index('-od') + 1]

    vids = make_vids(input_path,output_path)
    num_GPUs = 2
    processes = []
    indices = np.linspace(0, len(vids), num=num_GPUs + 1)

    # TODO: make this a cmd-line arg
    CONDA_ENV = '/home/emil/miniconda3/envs/br_doc/bin/python'

    for index in range(len(indices) - 1):
        if '-c' not in sys.argv:
            cmd = [
                CONDA_ENV,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'helpers', 'HalfCropper.py'), '-id', input_path, '-vl',
                str(int(indices[index])), '-vr',
                str(int(indices[index + 1])), '-od', output_path
            ]
        else:
            cmd = [
                CONDA_ENV,
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'helpers', 'HalfCropper.py'), '-id', input_path, '-od', output_path, '-vl',
                str(int(indices[index])), '-vr',
                str(int(indices[index + 1])), '-c', sys.argv[sys.argv.index('-c') + 1], '-n', sys.argv[sys.argv.index('-n') + 1]
            ]
        processes.append(
            subprocess.Popen(
                cmd, env={'CUDA_VISIBLE_DEVICES': '{0}'.format(str(index))}))
    start = timer()
    [p.wait() for p in processes]
    print(timer()-start, 'so viel zeit')
