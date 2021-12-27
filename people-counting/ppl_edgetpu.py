# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
# from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2



# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
help='Detection SSD model path (must have post-processing operator).')
parser.add_argument('--label', help='Labels file path.')
parser.add_argument('--input', type=str, help="path to optional input video file")
parser.add_argument('--output', help='Output image path.')
parser.add_argument('--keep_aspect_ratio', action='store_true',
help=(
  'keep the image aspect ratio when down-sampling the image by adding '
  'black pixel padding (zeros) on bottom or right. '
  'By default the image is resized and reshaped without cropping. This '
  'option should be the same as what is applied on input images during '
  'model training. Otherwise the accuracy may be affected and the '
  'bounding box of detection result may be stretched.'))
args = vars(parser.parse_args())

# Initialize engine.
engine = DetectionEngine(args["model"])
labels = dataset_utils.read_label_file(args["label"]) if args["label"] else None

vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = 640
H = 360

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalRight = 0
totalLeft = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    frame = vs.read()
    
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    # frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    # if we are supposed to be writing a video to disk, initialize
    # the writer
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % 20 == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        
        # Run inference.
        objs = engine.detect_with_image(frame, threshold=0.7, keep_aspect_ratio=args["keep_aspect_ratio"], relative_coord=False, top_k=5)

        # loop over the detections
        for obj in objs:
            # if the class label is not a person, ignore it
            if obj.label_id == 0:
            	# compute the (x, y)-coordinates of the bounding box
            	# for the object
            	box = obj.bounding_box.flatten().astype("int")
            	(startX, startY, endX, endY) = box
            	# construct a dlib rectangle object from the bounding
            	# box coordinates and then start the dlib correlation
            	# tracker
            	tracker = dlib.correlation_tracker()
            	rect = dlib.rectangle(startX, startY, endX, endY)
            	tracker.start_track(rgb, rect)
            	# add the tracker to our list of trackers so we can
            	# utilize it during skip frames
            	trackers.append(tracker)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    frame = np.uint8(frame)
    #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            x = [c[0] for c in to.centroids]
            direction = centroid[0] - np.mean(x)
            

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0:
                    totalLeft += 1
                    #log.write(time.strftime("%c") + '\t' + "Left" + '\t' + '\n')
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0:
                    totalRight += 1
                    #log.write(time.strftime("%c") + '\t' + "Right" + '\t' + '\n')
                    to.counted = True
            
            to.centroids.append(centroid)
                
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("left", totalLeft),
        ("right", totalRight),
        ("Status", status),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

#log.flush()
#log.close()

# check to see if we need to release the video writer pointer
if writer is not None:
    writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
    vs.stop()

# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()
