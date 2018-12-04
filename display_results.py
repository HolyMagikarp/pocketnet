import cv2
import os

results_dir = "detection_results/"
i = 0
for img_name in os.listdir(results_dir):
	img = cv2.imread(results_dir + img_name)
	cv2.imshow("detected", img)
	if i == 0:
		cv2.waitKey(0)
		i += 1
	cv2.waitKey(37)