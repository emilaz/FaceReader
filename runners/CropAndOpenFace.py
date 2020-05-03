"""
.. module CropAndOpenFace
    :synopsis: Contains a method for running OpenFace on a video directory as well as a class which crops a video and
    runs OpenFace on it
"""

import glob
import shutil
import json
import os
import subprocess
import sys
from dask import dataframe as df
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import VidCropper
from multiprocessing import Pool


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
         #'-fdir',
         #os.path.join(im_dir,'frames'),
         #'-of', 'au.csv', '-out_dir', im_dir,
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
        os.remove(os.path.join(im_dir, file_name+'.csv'))



class VideoImageCropper:

    def __init__(self,
                 vid=None,
                 im_dir=None,
                 already_cropped=None,
                 already_detected=None,
                 crop_txt_files=None,
                 nose_txt_files=None):

        self.already_cropped = already_cropped
        self.already_detected = already_detected
        self.im_dir = im_dir

        if not self.already_cropped and not self.already_detected:
            if crop_txt_files:
                self.crop_txt_files = crop_txt_files
            else:
                try:
                    self.crop_txt_files = json.load(
                        open(
                            os.path.join(
                                os.path.dirname(vid), 'crop_files_list.txt'),
                            mode='r'))
                except IOError:
                    self.crop_txt_files = find_txt_files(crop_path)

            if nose_txt_files:
                self.nose_txt_files = nose_txt_files
            else:
                try:
                    self.nose_txt_files = json.load(
                        open(
                            os.path.join(
                                os.path.dirname(vid), 'nose_files_list.txt'),
                            mode='r'))
                except IOError:
                    self.nose_txt_files = find_txt_files(nose_path)

            if not os.path.lexists(self.im_dir):
                os.mkdir(self.im_dir)

        VidCropper.CropVid(vid, self.im_dir, self.crop_txt_files,
                           self.nose_txt_files)


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
