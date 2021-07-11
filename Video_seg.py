import cv2
video_loc = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/camera_0.mp4"
img_loc = "/home/Mukherjee/Data/Fish_Tracking/5 fish video/Fish_img_side/"
vidcap = cv2.VideoCapture(video_loc)
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(img_loc+"image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.1 second
count=1
success = getFrame(sec)
#stop_time = count+1+500
while success:
    count = count + 1
    print(count)
    #if count == stop_time:
    #    break
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)