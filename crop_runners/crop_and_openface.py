"""
.. module crop_and_openface
    :synopsis: Contains a method for running OpenFace on a video directory as well as a class which crops a video and
    runs OpenFace on it
"""

import glob
import shutil
import pandas as pd
import os
import json
import subprocess
import sys
from dask import dataframe as df
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool

class DurationException(Exception):
    def __init__(self, string):
        Exception.__init__(self, string)


def duration(vid_file_path):
    ''' Video's duration in seconds, return a float number
    '''
    _json = probe(vid_file_path)

    if 'format' in _json:
        if 'duration' in _json['format']:
            return float(_json['format']['duration'])

    if 'streams' in _json:
        # commonly stream 0 is the video

        for s in _json['streams']:
            if 'duration' in s:
                return float(s['duration'])

    # if everything didn't happen,
    # we got here because no single 'return' in the above happen.
    raise DurationException('I found no duration for video {} given the cwd {}'.format(vid_file_path, os.getcwd()))
    # return None


def crop_and_resize(vid, directory,
                    resize_factor=5):
    """
    Crops a video and then resizes it

    :param vid: Video to crop
    :param width: Width of crop
    :type width: Union[int, float]
    :param height: Height of crop
    :type height: Union[int, float]
    :param x_min: x-coordinate of top-left corner
    :type x_min: Union[int, float]
    :param y_min: y-coordinate of top-left corner
    :type y_min: Union[int, float]
    :param directory: Directory to output files to
    :param resize_factor: Factor by which to resize the cropped video
    """
    patient, session, video = (os.path.splitext(os.path.basename(vid))[0]).split('_')
    path_to_crops = os.path.join('/home/emil/new_crop_coords/', patient, patient + '_' + session + '.hdf')
    crops = pd.read_hdf(path_to_crops)
    x_min, y_min, width, height = crops[crops['video'] == video]['coordinates'].values[0]
    subprocess.run(
        ['ionice', '-c2', '-n2',
         'ffmpeg', "-y", "-loglevel", "quiet",
         '-i', vid, '-vf', "crop={0}:{1}:{2}:{3}, scale={4}*iw:{4}*ih".format(
            str(width), str(height), str(x_min), str(y_min), str(resize_factor)),
         '-crf', '20',
         '-c:a', 'copy', os.path.join(directory, '%04d.png')], check=True)


def probe(vid_file_path):
    ''' Give a json from ffprobe command line

    :param vid_file_path : The absolute (full) path of the video file
    :type vid_file_path : str
    '''

    if type(vid_file_path) != str:
        raise Exception('Give ffprobe a full file path of the video')

    command = [
        "ffprobe", "-loglevel", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", vid_file_path
    ]

    pipe = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = pipe.communicate()
    out = out.decode()

    return json.loads(out)


def run_open_face(im_dirs, remove_intermediates=True) -> str:
    """
    Runs OpenFace
    :param im_dirs: List of locations of images
    :param remove_intermediates: Whether or not to remove intermediate files
    :return: void
    """
    executable = '/home/emil/OpenFace/build/bin/FeatureExtraction'  # Change to location of OpenFace
    temporary_result_folder = 'temporary'
    if not os.path.exists(temporary_result_folder):
        os.mkdir(temporary_result_folder)

    zipped = zip(['-fdir']*len(im_dirs),im_dirs)
    ready_for_command = [el for z in zipped for el in z]
    # try on images
    subprocess.run(
        ['ionice','-c2','-n2',
         executable, *ready_for_command,
         '-simsize', '5',
         '-out_dir', temporary_result_folder,
         '-wild','-multi_view', '1'],
        # stdout=subprocess.DEVNULL,
        check = True
        )

    if remove_intermediates:  # first step of removing intermediates
        for aligned in glob.glob(temporary_result_folder+'/*_aligned'):
            shutil.rmtree(aligned)
        for hog in glob.glob(temporary_result_folder+'/*hog'):
            os.remove(hog)
        for details in glob.glob(temporary_result_folder+'/*_details.txt'):
            os.remove(details)

        # new: also remove au_aligned and .hog file
        # shutil.rmtree(os.path.join(im_dir,'au_aligned'))
        # os.remove(os.path.join(im_dir,'au.hog'))
        # os.remove(os.path.join(im_dir,'au_of_details.txt'))
        # os.remove(os.path.join(im_dir,'au.csv'))

    args = zip(im_dirs, [temporary_result_folder] * len(im_dirs))
    pool = Pool(8)
    pool.starmap(move_and_clean, args)




def move_and_clean(im_dir, temporary_result_folder):
    # first, delete all those pics here
    for pic in glob.glob(im_dir + '/*.png'):
        os.remove(pic)

    file_name = os.path.basename(im_dir)
    vid_name = file_name.replace('_cropped', '')
    vid_name_parts = vid_name.split('_')
    patient_name = vid_name_parts[0]
    sess_num = vid_name_parts[1]
    vid_num = vid_name_parts[2]
    #move things from processed folder to where they need to be
    for file in glob.glob(os.path.join(temporary_result_folder,vid_name+'*')):
        shutil.move(file, im_dir)
    # this cleans the csv (unnecessary spaces and adds patietn/session/vid columns
    if file_name+'.csv' in os.listdir(im_dir):
        au_dataframe = df.read_csv(os.path.join(im_dir, file_name+'.csv'))
        # sLength = len(au_dataframe['frame'])
        au_dataframe = au_dataframe.assign(patient=lambda x: patient_name)
        au_dataframe = au_dataframe.assign(session=lambda x: sess_num)
        au_dataframe = au_dataframe.assign(video=lambda x: vid_num)
        au_dataframe[' success'] = au_dataframe[' confidence'] > .75  # set the limit for what we count as success higher
        au_dataframe = au_dataframe.compute()
        new_colnames = []

        for name in au_dataframe.columns:
            new_colnames.append(name.lstrip(' '))
        au_dataframe.columns = new_colnames

        # au_dataframe = au_dataframe.assign(
        df_dir = os.path.join(im_dir, 'hdfs')

        if not os.path.exists(df_dir):
            os.mkdir(df_dir)
        au_dataframe.to_hdf(os.path.join(df_dir, 'au.hdf'),key= '/data', format='table')
        # df.read_hdf(os.path.join(df_dir, 'au_0.hdf'), '/data') # assert saved correctly
        os.remove(os.path.join(im_dir, file_name+'.csv')) # delete the corresponding csv


#
# class VideoImageCropper:
#
#     def __init__(self,
#                  vid=None,
#                  im_dir=None,
#                  already_cropped=None,
#                  already_detected=None,
#                  crop_txt_files=None,
#                  nose_txt_files=None):
#
#         self.already_cropped = already_cropped
#         self.already_detected = already_detected
#         self.im_dir = im_dir
#
#         if not self.already_cropped and not self.already_detected:
#             if crop_txt_files:
#                 self.crop_txt_files = crop_txt_files
#             else:
#                 try:
#                     self.crop_txt_files = json.load(
#                         open(
#                             os.path.join(
#                                 os.path.dirname(vid), 'crop_files_list.txt'),
#                             mode='r'))
#                 except IOError:
#                     self.crop_txt_files = find_txt_files(crop_path)
#
#             if nose_txt_files:
#                 self.nose_txt_files = nose_txt_files
#             else:
#                 try:
#                     self.nose_txt_files = json.load(
#                         open(
#                             os.path.join(
#                                 os.path.dirname(vid), 'nose_files_list.txt'),
#                             mode='r'))
#                 except IOError:
#                     self.nose_txt_files = find_txt_files(nose_path)
#
#             if not os.path.lexists(self.im_dir):
#                 os.mkdir(self.im_dir)
#
#         VidCropper.CropVid(vid, self.im_dir, self.crop_txt_files,
#                            self.nose_txt_files)


def find_txt_files(path):
    return {
        os.path.splitext(os.path.basename(v))[0]: v

        for v in glob.iglob(os.path.join(path + '/**/*.txt'), recursive=True)
    }


if __name__ == '__main__':
    vid = None
    if '-v' in sys.argv:
        vid = sys.argv[sys.argv.index('-v') + 1]
    crop_path = sys.argv[sys.argv.index('-c') + 1]
    nose_path = sys.argv[sys.argv.index('-n') + 1]
    already_cropped = ('-ac' in sys.argv)
    already_detected = ('-ad' in sys.argv)

    if '-id' in sys.argv:
        im_dir = sys.argv[sys.argv.index('-id') + 1]
    else:
        im_dir = os.path.splitext(vid)[0] + '_cropped'
    crop = VideoImageCropper(
        vid,
        im_dir,
        crop_path,
        nose_path,
        already_cropped,
        already_detected)
