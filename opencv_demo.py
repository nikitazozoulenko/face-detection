from __future__ import print_function
from __future__ import division

from threading import Thread
import cv2
 
class WebcamVideoStream:
    def __init__(self, src=0):
	# initialize the video camera stream and read the first frame
	# from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, 1280)
        self.stream.set(4, 720)
        (self.grabbed, self.frame) = self.stream.read()
 
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
 
    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
	# indicate that the thread should be stopped
        self.stopped = True


# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
stream = WebcamVideoStream(src=0).start()
while True:
    frame = stream.read()
    # check to see if the frame should be displayed to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
cv2.destroyAllWindows()
vs.stop()
                























# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
    
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     print(ret)

#     # Our operations on the frame come here
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
