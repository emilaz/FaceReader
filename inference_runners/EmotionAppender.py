"""
.. module:: EmotionAppender
    :synopsis: Use an existing classifier to add classification data to a DataFrame
"""
import argparse
import pickle

import dask.dataframe as dd
import os

from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier

from dask.distributed import Client

import glob
from tqdm import tqdm

from inference_runners.EmotionPredictor import  get_prediction

def predict(df, classifier_function):
    data_columns = [x for x in df.columns if
                    x not in ['predicted','frame', 'face_id', 'success', 'timestamp', 'confidence', 'patient', 'video', 'annotated']]
    df = df[data_columns]
    predicted = classifier_function(df)
    return predicted


def add_classification(
    dataframe_path, classifier_path: RandomForestClassifier, emotion: str
):
    client = Client(processes=False)
    print(client)

    with parallel_backend("dask"):
        PATIENT_DIRS = [
            x
            for x in glob.glob(os.path.join(dataframe_path, "*cropped"))
            if "hdfs" in os.listdir(x)
               and 'Happy_predictions.hdf' not in os.listdir(os.path.join(x,'hdfs'))
        ]

        for patient_dir in tqdm(PATIENT_DIRS):
            try:
                curr_df = dd.read_hdf(
                    os.path.join(patient_dir, "hdfs", "au_w_anno.hdf"), "/data"
                )

                if (
                    len(curr_df)
                    and "annotated" in curr_df.columns
                    and "frame" in curr_df.columns
                ):
                    inference_columns = [x for x in curr_df.columns if
                                    x not in ['predicted', 'frame', 'face_id', 'success', 'timestamp', 'confidence',
                                              'patient', 'video', 'annotated', 'session']]
                    inference_df = curr_df[inference_columns].compute()
                    happy = get_prediction(classifier_path,inference_df,0.4862) #found during training
                    imp_columns=['patient','session','video','frame','success','confidence','annotated']
                    emotion_df = curr_df[imp_columns].compute()
                    emotion_df['Happy'] = happy

                    # store in the out_fullpath
                    emotion_df.to_hdf(
                        os.path.join(patient_dir,'hdfs',"{0}_predictions.hdf".format(emotion)),
                        "/data",
                        format="table",
                        scheduler="processes",
                    )
                else:
                    print(patient_dir + "HAS A PROBLEM")

            except AttributeError as e:
                print(e)
            except ValueError as e:
                print(e)
            except KeyError as e:
                print(e)

        # dataframe = dataframe.compute().assign(
        # predicted=lambda x: predict(x, classifier_path)
        # )
    # dataframe.to_hdf(dataframe_path, "/data", format="table", scheduler="processes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", help="Path to DataFrame containing folder")
    parser.add_argument("-classifier", help="Path to classifier")
    parser.add_argument("-emotion", help="Emotion")
    #parser.add_argument("out_subfolder", help="Sub folder to store the emotion predictions (it will be stored under dataframe_folder directory)")


    ARGS = parser.parse_args()
    classifier_path = pickle.load(open(ARGS.classifier, "rb"))
    print(classifier_path)

    emotion = ARGS.emotion

    add_classification(ARGS.data, classifier_path, emotion)
