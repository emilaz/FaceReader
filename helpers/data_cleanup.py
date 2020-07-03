"""
This module is needed for deleting of unnecessary files that were created during the FaceReader pipeline, but
take up significant amounts of space. These include 98% of .avi files (some left for insight) and au_w_anno.hdf
"""
import glob
import os
import sys

all_avi_files = glob.glob('/data1/users/emil/openface_output/**/*.avi', recursive=True)
if len(all_avi_files)>150:  # safeguard to avoid deleting twice. We want to keep a minimum number.
    [os.remove(f) for idx, f in enumerate(all_avi_files) if idx % 50 != 0]
print('Deleted .avi files')

# delete au_w_anno
aus_w_anno = glob.glob('/data1/users/emil/openface_output/**/hdfs/au_w_anno.hdf')
[os.remove(a) for a in aus_w_anno]

print('Done')
sys.exit(0)