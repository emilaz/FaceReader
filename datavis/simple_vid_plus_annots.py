from datavis.HappyVidMarker import bar_movie
import pandas as pd
import sys


if __name__ == '__main__':
    """Input: 
    Path to video (-vid) 
    Path to what you want to have plotted (-bar). You might need to change code depending on what you want to plot
    Path to output dir (-od)
    
    Output:
    Your video with the plot above, including a progress line.
    """

    vid = sys.argv[sys.argv.index('-vid') + 1]
    data = sys.argv[sys.argv.index('-bar') + 1]
    out = sys.argv[sys.argv.index('-od') + 1]
    df = pd.read_hdf('../openface_test/cb46fd46_5_0074_cropped/hdfs/au_w_anno.hdf')
    marker = df['annotated']
    time = df['frame']
    bar_movie(vid,out,time,marker,False)