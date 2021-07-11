import os
from os.path import isfile, join
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def blob_detect(img_loc):

    #img_loc = "/home/Mukherjee/Data/Fish Tracking/Fish_img/image10.jpg"

    img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape
    old_img = img
    #print(img.shape)
    img = img[127:2600, :]
    # plt.imshow(img)
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -1, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1],
                         flags=cv2.INTER_LINEAR)
    kernel = np.ones((5, 5), dtype=np.float32)/25
    img = cv2.filter2D(img, -1, kernel)
    #img = cv2.GaussianBlur(img,(9,9),0)
    img = cv2.medianBlur(img,9)


    #detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 23  # 30
    params.maxThreshold = 100  # 100

    params.filterByArea = True
    params.minArea = 30  # 240
    params.maxArea = 10000  # 10000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 0.9

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0
    params.maxConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    num = len(keypoints)
    for i in range(num):
        kp_list = list(keypoints[i].pt)
        kp_list[1] = kp_list[1]+127
        if keypoints[i].pt[0]< 679 and keypoints[i].pt[0]>681 and keypoints[i].pt[1]<2260 and keypoints[i].pt[1]>2262:
            keypoints[i].pt = tuple(kp_list)
        print(list(keypoints[i].pt))

    #print(keypoints[0].response)

    # print(keypoints)
    img_with_keypoints = cv2.drawKeypoints(old_img, keypoints, np.array([]),
                                           (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print(img_with_keypoints.shape)

    return img_with_keypoints, num


def blob_vid_create(path_In, path_Out, fps):
    frame_array = []
    files = [f for f in os.listdir(path_In) if isfile(join(path_In, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    #files.sort()

    count_one = 0
    count_more = 0
    count_zero = 0
    bad_frame = []
    for i in tqdm(range(len(files))):
        filename = path_In + files[i]
    # reading each files
        img, num = blob_detect(filename)
        height, width, layers = img.shape
        size = (width, height)
        if num == 1:
            count_one = count_one+1
        elif num > 1:
            count_more = count_more+1
            bad_frame.append(files[i])
            bad_frame.append(num)
        else:
            count_zero = count_zero+1
            bad_frame.append(files[i])

    # inserting the frames into an image array
        frame_array.append(img)
    per1 = (count_zero/len(files))*100
    per2 = (count_one/len(files))*100
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


img_loca = "/home/Mukherjee/Data/Fish_Tracking/Fish_img_top/image17.jpg"
img, num = blob_detect(img_loca)
print(num)
plt.imshow(img)
vid_out = "/home/Mukherjee/Data/Fish_Tracking/vid_top.avi"
img_loc = "/home/Mukherjee/Data/Fish_Tracking/Fish_img_top/"
fps = 2
#blob_vid_create(img_loc,vid_out,fps)
