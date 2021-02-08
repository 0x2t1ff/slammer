#!/usr/bin/env python3
import numpy as np
import cv2
from detector import KeypointDetector

class VideoSlammer():
  
  def __init__(self, path):
    self.cap = cv2.VideoCapture('car.mp4')
    self.kd = KeypointDetector()

  
  def start(self):
    while self.cap.isOpened():

      ret, frame = self.cap.read()
      frame = cv2.resize(frame, (1920//2, 1080//2))
      features = self.kd.detect(frame)
      self.kd.drawKeypoints(frame, features)
      self.kd.compute(frame)
      self.kd.compare()

      cv2.imshow('car', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

slam = VideoSlammer('test.mp4')

slam.start()
