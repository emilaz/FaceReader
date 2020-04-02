from tqdm import tqdm
from typing import List
from dask import dataframe as df
from dask import array as da
import glob
from pathos.multiprocessing import ProcessingPool as Pool
import functools
import sys
import os
from helpers.patient_info import patient_day_session, get_patient_names
from runners import VidCropper
import AUGui

"""
.. module:: AUvsEmotionGenerator
    :synopsis: Appends classified emotion to AU dataframe
"""


def clean_to_write(to_write: str) -> str:
    if to_write == 'Surprised':
        to_write = 'Surprise'
    elif to_write == 'Disgusted':
        to_write = 'Disgust'
    elif to_write == 'Afraid':
        to_write = 'Fear'

    return to_write


def find_scores(patient_dir: str, refresh: bool):
    """
    Finds the scores for a specific patient directory

    :param patient_dir: Directory to look in
    """

    if not refresh and 'au_w_anno.hdf' in os.listdir(os.path.join(patient_dir,'hdfs')):
        return

    try:
        patient, day, session = patient_day_session(patient_dir)
        try:
            au_frame = df.read_hdf(
                os.path.join(patient_dir, 'hdfs', 'au.hdf'), '/data')
        except ValueError as e:
            print(e)

        # except ValueError as e:
            # print(e)

            # return

        if 'frame' not in au_frame.columns:
            return

        annotated_values = ["N/A" for _ in range(len(au_frame.index))]

        #here are the hand annotations
        csv_path = os.path.join('/home/emil/emotion_annotations',patient_dir.replace('cropped','emotions.csv'))
        num_frames = int(
            #VidCropper.duration(get_vid_from_dir(patient_dir)) * 30) #this is the length of orig video
            VidCropper.duration(os.path.join(patient_dir,'au.avi')) * 30)  # this is the length of au video (should be 1 frame less than orig


        if os.path.exists(csv_path):
            csv_dict = AUGui.csv_emotion_reader(csv_path)

            if csv_dict:
                annotated_ratio = int(num_frames / len(csv_dict))
                if annotated_ratio>1:
                    print('HELLO HERE IS SUCH A CASE:',patient_dir)
                if annotated_ratio == 0:
                    annotated_ratio = 1
                csv_dict = {
                    i * annotated_ratio: c

                    for i, c in csv_dict.items()
                }

                for i in [
                        x for x in csv_dict.keys() if 'None' not in csv_dict[x]
                ]:
                        to_write = clean_to_write(csv_dict[i])

                        if i in range(len(annotated_values)):
                            annotated_values[i] = to_write
        # au_frame = au_frame.assign(annotated=annotated_values)
        # au_frame = au_frame.set_index('frame')
        # au_frame["annotated"] = df.from_array(da.from_array(annotated_values, chunks=5))
        annotated_values = da.from_array(annotated_values, chunks='auto').compute()

        #what we know: au_frame['frame'] starts at 1, goes to (including) 3604
        #annotated_values has length we want, but currently (with the +1) a length of 3605
        #au_frame has a length of 3604 (makes sense, 1-3604)

        au_frame = au_frame.compute()
        au_frame = au_frame.assign(annotated=lambda x: annotated_values[x['frame'] - 1])

        au_frame.to_hdf(
            os.path.join(patient_dir, 'hdfs', 'au_w_anno.hdf'),
            '/data',
            format='table')

    except FileNotFoundError as not_found_error:
        print(not_found_error)

    except AttributeError as e:
        print(e)


def find_one_patient_scores(patient_dirs: List[str], refresh: bool, patient: tuple):
    """Finds the annotated emotions for a single patient and adds to overall patient DataFrame.

    :param patient_dirs: All directories ran through OpenFace.
    :param patient: Patient to find annotated emotions for
    """
    tqdm_position, patient = patient
    curr_dirs = [x for x in patient_dirs if patient in x]

    for patient_dir in tqdm(curr_dirs, position=tqdm_position):
        find_scores(patient_dir, refresh)


if __name__ == '__main__':
    OPEN_DIR = sys.argv[sys.argv.index('-d') + 1]
    refresh = '--refresh' in sys.argv
    os.chdir(OPEN_DIR)
    # Directories have been previously cropped by CropAndOpenFace
    PATIENT_DIRS = [
        x for x in glob.glob('*cropped') if 'hdfs' in os.listdir(x)
                                            and 'au_w_anno.hdf' not in os.listdir(os.path.join(x,'hdfs'))
    ]
    #this gets their names_sess_vid, no path or extension or anything.
    PATIENTS = get_patient_names(PATIENT_DIRS)

    # find_one_patient_scores(PATIENT_DIRS,refresh,(0,PATIENTS[0]))

    ######
    PARTIAL_FIND_FUNC = functools.partial(find_one_patient_scores, PATIENT_DIRS, refresh)
    TUPLE_PATIENTS = [((i % 5), x) for i, x in enumerate(PATIENTS)]
    Pool(5).map(PARTIAL_FIND_FUNC, TUPLE_PATIENTS)
    #######

    # Pool().map(find_scores, PATIENTS)
    # for i, x in enumerate(PATIENTS):
        # tuple_patient = (i % cpu_count(), x)
        # find_one_patient_scores(PATIENT_DIRS, tuple_patient)
