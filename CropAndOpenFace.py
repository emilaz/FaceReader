import glob
import json
import os
import subprocess
import sys
import numpy as np

sys.path.append('/home/gvelchuru')
from OpenFaceScripts import ImageCropper, VidCropper


def run_open_face(im_dir, vid_mode = False):
    executable = '/home/gvelchuru/OpenFace/build/bin/FeatureExtraction'  # Change to location of OpenFace
    if not vid_mode:
        subprocess.Popen("ffmpeg -y -r 30 -f image2 -s 1920x1080 -pattern_type glob -i '{0}' -b:v 2000k {1}".format(
            os.path.join(im_dir, '*.png'),
            os.path.join(im_dir,
                         'inter_out.mp4')), shell=True).wait()
        vid_name = 'inter_out.mp4'
    else:
        vid_name = 'inter_out.avi'

    # Remove q if visualization desired, inserted for performance
    subprocess.Popen(
        '{0} -f {1} -of {2} -q -verbose -wild -multi-view 1'.format(executable,
                                                                            os.path.join(im_dir,
                                                                                         vid_name),
                                                                            os.path.join(im_dir, 'au.txt')),
        shell=True).wait()
    os.remove(os.path.join(im_dir, vid_name))


class VideoImageCropper:
    def __init__(self, vid=None, im_dir=None, crop_path=None, nose_path=None, already_cropped=None,
                 already_detected=None, crop_txt_files=None, nose_txt_files = None, vid_mode = False):
        self.already_cropped = already_cropped
        self.already_detected = already_detected
        self.im_dir = im_dir
        self.crop_path = crop_path
        self.nose_path = nose_path
        if not self.already_cropped and not self.already_detected:
            if crop_txt_files:
                self.crop_txt_files = crop_txt_files
            else:
                try:
                    self.crop_txt_files = json.load(
                        open(os.path.join(os.path.dirname(vid), 'crop_files_list.txt'), mode='r'))
                except IOError:
                    self.crop_txt_files = find_txt_files(crop_path)
            if nose_txt_files:
                self.nose_txt_files = nose_txt_files
            else:
                try:
                    self.nose_txt_files = json.load(
                        open(os.path.join(os.path.dirname(vid), 'nose_files_list.txt'), mode='r'))
                except IOError:
                    self.nose_txt_files = find_txt_files(nose_path)
            if not os.path.lexists(self.im_dir):
                os.mkdir(self.im_dir)
        if not vid_mode:
            subprocess.Popen('ffmpeg -y -i "{0}" -vf fps=30 "{1}"'.format(vid, os.path.join(self.im_dir, (
                os.path.basename(vid) + '_out%04d.png'))), shell=True).wait()
            ImageCropper.CropImages(self.im_dir, self.crop_txt_files, self.nose_txt_files, save=True)
            if len(glob.glob(os.path.join(self.im_dir, '*.png'))) > 0:
                if not self.already_detected:
                    run_open_face(self.im_dir)
                frame_direc = os.path.join(self.im_dir, 'labeled_frames/')
                if not os.path.exists(frame_direc):
                    os.mkdir(os.path.join(self.im_dir, 'labeled_frames/'))
                subprocess.Popen(
                    'ffmpeg -y -i "{0}" -vf fps=30 "{1}"'.format(os.path.join(self.im_dir, 'out.mp4'), os.path.join(frame_direc, (
                        os.path.basename(self.im_dir) + '_out%04d.png'))), shell=True).wait()
        else:
            VidCropper.CropVid(vid, self.im_dir, self.crop_txt_files, self.nose_txt_files)
            run_open_face(self.im_dir, vid_mode=True)


def find_txt_files(path):
    return {os.path.splitext(os.path.basename(v))[0]: v for v in
            glob.iglob(os.path.join(path + '/**/*.txt'), recursive=True)}


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
    crop = VideoImageCropper(vid, im_dir, crop_path, nose_path, already_cropped, already_detected, vid_mode=True)
