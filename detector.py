import cv2
import math
import numpy as np
class KeypointDetector():

  def __init__(self, maxDistance=100):
    self.orb = cv2.ORB_create()
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    self.last_des = None
    self.maxDistance = maxDistance

  def detect(self, img):
    # Gray scale the image to run goodFeaturesToTrack
    # Not sure if it's better to track more simplistic features (as here?)
    # than maybe using other more modern extractors that may return fewer kps
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    self.corners = cv2.goodFeaturesToTrack(grayed, 1000, 0.01, 3)
    return self.corners

  def compute(self, img):
    # Compute given corners with ORB
    # In order to compute them, create KeyPoints object from x,y
    # coordinates the corners extracted
    kps = [cv2.KeyPoint(*p[0], _size=512) for p in self.corners]
    kps, des = self.orb.compute(img, kps)
    self.kps = kps
    self.des = des
    self.img = img

  def compare(self):
    # Compare with latest computed kp/des
    good = []
    if self.last_des is not None:
      matches = self.bf.knnMatch(self.des, self.last_des, k=2)
      for m,n in matches:
        if m.distance < 0.7*n.distance:
          pt1 = tuple(map(lambda x:int(x), self.last_kps[m.trainIdx].pt))
          pt2 = tuple(map(lambda x:int(x), self.kps[m.queryIdx].pt))
          # Poopoo way to remove wrong matches with distance between points
          # being too high. Good old pythagoras
          distance = math.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
          if distance < self.maxDistance:
            cv2.line(self.img, pt1, pt2, (255, 0,0), 1)
            good.append(m)
    self.last_des = self.des
    self.last_kps = self.kps
    
    
  def drawKeypoints(self, img, f):
    self.img = img
    for i in f:
      x, y = map(lambda x:int(x), i.ravel())
      cv2.circle(img, (x,y), 2, (0,255,0), -1)
      cv2.circle(img, (100,100), 2, (0,255,0), -1)

