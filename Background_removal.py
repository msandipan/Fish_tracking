# read 100 frames into here
import os
from os.path import isfile, join
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_frames(frame_loc,num):

    frame_array = []
    files = [f for f in os.listdir(frame_loc) if isfile(join(frame_loc, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    #files.sort()
    for i in range(len(files)):
        if i >= num:
            break
        filename = frame_loc + files[i]
        #rint(filename)
        img = cv2.imread(filename)
        #img =
        frame_array.append(img)

    return frame_array

def read_frames_vid(vid_loc,num):
    video_stream = cv2.VideoCapture(vid_loc)

# Randomly select 30 frames
    frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=num)

    # Store selected frames in an array
    frames = []
    for fid in tqdm(frameIds):
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)

    video_stream.release()
    return frames

def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

loc = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/Fish_img_side/"
frames = read_frames(loc,50)
print(len(frames))

def get_median(frames):
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    #plt.imshow(fixColor(medianFrame))
    return medianFrame

def remove_bck(sample,medianFrame):
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    graySample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(graySample, grayMedianFrame)

    plt.imshow(dframe,cmap='gray')
    return dframe
medianFrame = get_median(frames)
remove_bck(frames[1],medianFrame)