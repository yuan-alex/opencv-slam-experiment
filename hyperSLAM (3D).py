import datetime

print("hyperSLAM.py %s" %datetime.datetime.now())
print("A New Simple Framework for Simultaneous Localization and Mapping")
print("Developed by: Alex Yuan")
print("Starting...\n")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

print("Packages Imported...\n")

vid_dir = "drive.mp4"
drive_video = cv2.VideoCapture(vid_dir)

first_run = True
store_frame_counter = 8

z_counter = 1

# Does all the actual SLAM processing
class MatchKeypoints(object):
    def __init__(self, prev = {"kp": None, "des": None, "frame": None, "z": None}, bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True), matches = None):
        # Storing all previous stuffs in a dict
        self.prev = prev
        self.bf = bf
        self.matches = matches

    def store_as_previous(self, kp, des, z, frame):

        # Just stores the previous keypoints and descriptors for later processing
        # Storing frame as well because why not, nothing much
        self.prev["kp"] = kp
        self.prev["des"] = des
        self.prev["z"] = z
        self.prev["frame"] = frame

    def match_keypoints(self, current_des):
        # Should automaticlly match the current des to the previous des
        self.matches = self.bf.match(current_des, self.prev["des"])

        # Sort them in the order of their distance
        self.matches = sorted(self.matches, key = lambda x:x.distance)

        return self.matches

# Does all the ORB extraction
class FeatureExtractor(object):
    def __init__(self, kp = None, des = None, orb = cv2.ORB_create(nfeatures = 3500)):
        self.orb = orb
        self.kp = kp
        self.des = des

    def get_features(self, frame):

        # Had to replace it with detectAndCompute; the docs are really confusing
        # It only works with compute for some reason
        self.kp, self.des = self.orb.detectAndCompute(frame, None)
        return self.kp, self.des

# Does everything related to drawing the points
class Draw(object):
    def __init__(self, custom_color = (0, 255, 0), new_frame = None, font = cv2.FONT_HERSHEY_SIMPLEX,
                current_frame_idx = None, prev_frame_idx = None, x1 = None, x2 = None, y1 = None, y2 = None,
                plot_x1 = None, plot_x2 = None):
        self.custom_color = custom_color
        self.new_frame = new_frame

        self.current_frame_idx = current_frame_idx
        self.prev_frame_idx = prev_frame_idx
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.font = font

        self.plot_x1 = plot_x1
        self.plot_x2 = plot_x2

    def keypoints(self, frame, keypoints):
        self.new_frame = frame.copy()

        # There are too many keypoints to draw and it laggs behind.
        '''
        cv2.drawKeypoints(self.new_frame, keypoints, None, color = self.custom_color, flags = 0)

        show_frame = True
        for keypoint in keypoints[:int(len(keypoints)/2)]:
            if True:
                cv2.putText(self.new_frame, "hyperSLAM.py", (10, 50), self.font, 2, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(self.new_frame, "A New Simple Framework for Simultaneous Localization and Mapping", (10, 95), self.font, 1, (255,255,255), 1, cv2.LINE_AA)

                cv2.putText(self.new_frame, "KEYPOINT", (int(keypoint.pt[0]) - 20, int(keypoint.pt[1]) - 20), self.font, 0.2, (255,255,255), 1, cv2.LINE_AA)
                cv2.circle(self.new_frame, (int(keypoint.pt[0]), int(keypoint.pt[1])), 4, (0, 255, 0), 1)

                show_frame = False
            else:
                show_frame = True

        cv2.imshow("orb", self.new_frame)
        '''
        return self.new_frame

    def matches(self, current_frame, prev_frame, matches, current_kp, prev_kp, current_des, prev_des):
        self.new_frame = current_frame.copy()

        cv2.putText(self.new_frame, "hyperSLAM.py", (10, 50), self.font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self.new_frame, "A New Simple Framework for Simultaneous Localization and Mapping", (10, 95), self.font, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(self.new_frame, "Made by: Alex Yuan", (10, 120), self.font, 0.8, (255,255,255), 1, cv2.LINE_AA)

        self.plot_x1 = []
        self.plot_y1 = []

		# Only get the top 100 best matches. The matches list should be already sorted.
        for match in matches[:100]:
            # Get the matches which has a distance below or equal to 50.
            if match.distance <= 50:

                # NOTE: DO NOT GET CONFUSED BETWEEN QUERYIDX AND TRAINIDX
                self.current_frame_idx = match.queryIdx
                self.prev_frame_idx = match.trainIdx

                (self.x1, self.y1) = prev_kp[self.prev_frame_idx].pt
                (self.x2, self.y2) = current_kp[self.current_frame_idx].pt

                cv2.circle(self.new_frame, (int(self.x1), int(self.y1)), 2, (255,0,0), 1)
                cv2.circle(self.new_frame, (int(self.x2), int(self.y2)), 1, self.custom_color, 1)
                cv2.line(self.new_frame, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), self.custom_color, 1)

                cv2.putText(self.new_frame, "SLAM MOVEMENT", (int(self.x1) - 50, int(self.y1) - 20), self.font, 0.4, (255,255,255), 1, cv2.LINE_AA)

                norm_x1 = [float(i)/max(self.plot_x1) for i in self.plot_x1[:2]]
                norm_y1 = [float(i)/max(self.plot_y1) for i in self.plot_y1[:2]]

                if match.distance <= 10:
                    self.plot_x1.append(self.x1)
                    self.plot_y1.append(self.y1)
                    ax.scatter(norm_x1, z_counter, norm_y1, c='blue', marker='o')

        return self.new_frame

Draw = Draw()
FeatureExtractor = FeatureExtractor()
MatchKeypoints = MatchKeypoints()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

#for i in range(10,77):
while True:
    # Read the image from the frame
    _, frame = drive_video.read()
    #frame = cv2.imread("data/00000000%s.png" %i)


    # Get the features of the current frame
    kp, des = FeatureExtractor.get_features(frame)

    if not first_run:
        # The reason why this is because don't expect there to be a previous frame
        # the first time you run the program

        # Get the matches from the current and the previous frame
        matches = MatchKeypoints.match_keypoints(des)

        orb_frame = Draw.keypoints(frame, kp)

        # Draw the matches
        matches_frame = Draw.matches(frame, MatchKeypoints.prev["frame"], matches, kp, MatchKeypoints.prev["kp"], des, MatchKeypoints.prev["des"])
        cv2.imshow("matches_frame", matches_frame)

    if store_frame_counter % 8 == 0:


        # Stored as previous at the very end of all the processing
        MatchKeypoints.store_as_previous(kp, des, z_counter, frame)
        cv2.imshow("old", MatchKeypoints.prev["frame"])
        store_frame_counter = 1
        z_counter = z_counter + 1

    else:
        store_frame_counter = store_frame_counter + 1

    press_char = cv2.waitKey(1)

    if press_char == ord("q"):
        break

    if press_char == ord("p"):
        plt.show()

    elif first_run:
        first_run = False


drive_video.release()
cv2.destroyAllWindows()
