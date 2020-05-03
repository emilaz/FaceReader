import os
import sys
import subprocess  # TODO get rid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from runners.MultiCropAndOpenFace import make_vids, make_crop_and_nose_files
from runners import CropAndOpenFace, VidCropper
from runners.VidCropper import DurationException
from tqdm import tqdm


def run():
    for vid in tqdm(range(vid_left, vid_right)):
        crop_image(vid)


def crop_image(vid, output_path):
    # vid = vids[i]
    #im_dir = os.path.splitext(vid)[0] + '_cropped'
    im_dir = os.path.join(output_path,os.path.splitext(vid)[0].split('/')[-1] + '_cropped')
    try_cropping(vid, im_dir)


def try_cropping(vid, im_dir):
    try:
        VidCropper.duration(vid)
        print('in HalfCropper:', subprocess.run(['ffmpeg','-version'],check=True,stdout=subprocess.PIPE))
        CropAndOpenFace.VideoImageCropper(
            vid=vid,
            im_dir=im_dir,
            crop_txt_files=crop_txt_files,
            nose_txt_files=nose_txt_files)
    except DurationException as e:
        print(str(e) + '\t' + vid)


# if __name__ == '__main__':
#     input_path = sys.argv[sys.argv.index('-id') + 1]
#     output_path = sys.argv[sys.argv.index('-od') + 1]
#
#     crop_txt_files, nose_txt_files = make_crop_and_nose_files(output_path)
#
#     os.chdir(output_path)
#     vids = make_vids(input_path, output_path)
#     vid_left = int(sys.argv[sys.argv.index('-vl') + 1])
#     vid_right = int(sys.argv[sys.argv.index('-vr') + 1])
#     run()
