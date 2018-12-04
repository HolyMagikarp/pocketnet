import cv2
import sys

############################################################################################################################################
# 1. Once a frame pops up, draw a rectangle over the object you would like to track, then press Enter.
# 2. Repeat once.
# 3. Watch.
############################################################################################################################################

video = cv2.VideoCapture("pokemon.mp4")
success, frame = video.read()

if not success:
    print("read failed")
else:
    count = 1
    while count <= 21680:
        success, frame = video.read()
        count += 1
    success, frame = video.read()
    # select one pokemon
    box1 = cv2.selectROI(frame, False)
    # select the other
    box2 = cv2.selectROI(frame, False)
    
    tracker1 = cv2.TrackerKCF_create()
    tracker2 = cv2.TrackerKCF_create()
#     tracker1 = cv2.TrackerTLD_create()
#     tracker2 = cv2.TrackerTLD_create()
#     tracker1 = cv2.TrackerMedianFlow_create()
#     tracker2 = cv2.TrackerMedianFlow_create()
    
    tracker1.init(frame, box1)
    tracker2.init(frame, box2)
    
    success, frame = video.read()
    i = 0
    while success:
        success1, box1 = tracker1.update(frame)
        success2, box2 = tracker2.update(frame)
        
        if success1:
            top_left = (int(box1[0]), int(box1[1]))
            bottom_right = (int(box1[0] + box1[2]), int(box1[1] + box1[3]))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), thickness=3)
        else:
            cv2.putText(frame, "Tracker 1 did not find a matching object", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=2)
        
        if success2:
            top_left = (int(box2[0]), int(box2[1]))
            bottom_right = (int(box2[0] + box2[2]), int(box2[1] + box2[3]))
            cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), thickness=3)
        else:
            cv2.putText(frame, "Tracker 2 did not find a matching object", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), thickness=2)
            
        cv2.imshow("tracking", frame)
        if i == 0 :
            cv2.waitKey(0)
            i += 1
        else:
            cv2.waitKey(37)
        
        success, frame = video.read()