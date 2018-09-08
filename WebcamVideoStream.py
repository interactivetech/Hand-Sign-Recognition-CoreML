# import the necessary packages
from threading import Thread
import sys
import cv2

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue

# import the necessary packages
from threading import Thread
import cv2

class WebcamVideoStream:
	def __init__(self, src=0):
            # initialize the video camera stream and read the first frame
            # from the stream
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,416)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,416)
            (self.grabbed, self.frame) = self.stream.read()
			# self.frame = cv2.resize(self.frame,dsize=(224,224))

            # initialize the variable used to indicate if the thread should
            # be stopped
            self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			# self.frame = cv2.resize(self.frame,dsize=(224,224))

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True