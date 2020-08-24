# FaceRuner
This package is designed to infer emotions from video data. 
It uses OpenFace to extract facial features and subsequently applies a pretrained classifier on those features.
So far, only a trained classifier for happiness is available, but train routines are available in inference_runners/train_classifier.py

Workflow is as follows:
1. Run crop_runners/multi_crop_and_openface.py on directory with videos in question
2. Run inference_runners/annotation_appender.py on the output dir. This adds a annotation column to the output dataframe of step 1, needed for consistency reasons
3. Run inference_runnres/inferenced_emotion_appender.py to infer emotions and add to dataframe.