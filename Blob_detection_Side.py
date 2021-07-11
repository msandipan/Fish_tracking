import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
from tqdm import tqdm
from scipy.spatial.distance import euclidean


def blob_detect(img_loc):

    #img_loc = "/home/Mukherjee/Data/Fish Tracking/Fish_img/image10.jpg"

    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    old_img = img
    img_crop = img[575:width, :]
    # plt.imshow(img_crop)
    image_center = tuple(np.array(img_crop.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -1, 1.0)
    img = cv2.warpAffine(img_crop, rot_mat, img_crop.shape[1::-1],
                         flags=cv2.INTER_LINEAR)
    kernel = np.ones((5, 5), dtype=np.float32)/25
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.medianBlur(img,9)

    #detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 115  # 130
    params.maxThreshold = 175  # 170

    params.filterByArea = True
    params.minArea = 100  # 160
    params.maxArea = 4000  # 4000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.25

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    num = len(keypoints)
    for i in range(num):
        kp_list = list(keypoints[i].pt)
        kp_list[1] = kp_list[1]+575
        keypoints[i].pt = tuple(kp_list)
    #euc_dist = []
    #for i in range(num):
    #    kp_list1 = np.array(list(keypoints[i].pt))
    #    if i<(num-1):
    #        j = i+1
    #        kp_list2 = np.array(list(keypoints[j].pt))
    #    else:
    #        j = 0
    #        kp_list2 = np.array(list(keypoints[j].pt))
    #    dist = euclidean(kp_list1,kp_list2)
    #    euc_dist.append([dist,(i,j)])
    #    print(dist)
    #for i in range(len(euc_dist)):
    #    dist,pair = euc_dist
    #    if dist<15:
    #        size1 =  keypoints[pair[0]].size
    #        size2 = keypoints[pair[1]].size
    #        if size1>size2:

    #print(keypoints[0].size)
    #print(keypoints[1].size)
    # print(keypoints)
    img_with_keypoints = cv2.drawKeypoints(old_img, keypoints, np.array([]),
                                           (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints, num


def blob_vid_create(path_In, path_Out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_In) if isfile(join(path_In, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    files.sort()

    count_five = 0

    count_more = 0
    count_less = 0
    bad_frame = []
    for i in tqdm(range(len(files))):
        filename = path_In + files[i]
    # reading each files
        img, num = blob_detect(filename)
        height, width, layers = img.shape
        size = (width, height)
        if num == 5:
            count_five = count_five+1
        elif num > 5:
            count_more = count_more+1
            bad_frame.append(files[i])
            bad_frame.append(num)

        else:
            count_less = count_less+1
            bad_frame.append(files[i])
            bad_frame.append(num)

    # inserting the frames into an image array
        frame_array.append(img)
    per1 = (count_less/len(files))*100
    per2 = (count_five/len(files))*100
    per3 = (count_more/len(files))*100
    print(per1, per2, per3)
    print(bad_frame)
    print(len(files))
    out = cv2.VideoWriter(path_Out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    print("Done")


img_loca = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/Fish_img_side/image10.jpg"
img, num = blob_detect(img_loca)
print(num)
plt.imshow(img)
vid_out = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/vid_side.avi"
img_loc = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/Fish_img_side/"
fps = 2
#blob_vid_create(img_loc,vid_out,fps)
