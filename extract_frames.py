import cv2
import os

# HAVE the video pokemon.mp4 in the same directory

vidcap = cv2.VideoCapture('pokemon.mp4')
success,image = vidcap.read()
count = 0
success = True

os.makedirs("frames", exist_ok=True)
while success:
    success,image = vidcap.read()
    ### take frame 16751 to frame 23678
    if count >= 16751:
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    count += 1
    if count > 23678:
        break