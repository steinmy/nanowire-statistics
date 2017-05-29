import numpy as np
import cv2

def detect(image,
           invert,
           minThreshold=0,
           maxThreshold=255,
           thresholdStep=1,
           minDistBetweenBlobs=0,
           filterByArea=False,
           minArea=0,
           maxArea=None,
           filterByCircularity=False,
           minCircularity=0.0,
           maxCircularity=None,
           filterByConvexity=False,
           minConvexity=0.0,
           maxConvexity=None,
           filterByInertia=False,
           minInertiaRatio=0.0,
           maxInertiaRatio=None
           ):

    if invert:
        image = 255 - image

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = minThreshold
    params.maxThreshold = maxThreshold
    params.thresholdStep = thresholdStep
    params.minDistBetweenBlobs = minDistBetweenBlobs
    params.filterByArea = filterByArea
    params.minArea = minArea
    params.maxArea = maxArea
    params.filterByCircularity = filterByCircularity
    params.minCircularity = minCircularity
    params.maxCircularity = maxCircularity
    params.filterByConvexity = filterByConvexity
    params.minConvexity = minConvexity
    params.maxConvexity = maxConvexity
    params.filterByInertia = filterByInertia
    params.minInertiaRatio = minInertiaRatio
    params.maxInertiaRatio = maxInertiaRatio

    # Set up the detector with the given parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    # Do the detection
    keypoints = detector.detect(image)

    blobs = np.zeros((len(keypoints), 3))

    for i, keypoint in enumerate(keypoints):
        blobs[i][0] = keypoint.pt[1]
        blobs[i][1] = keypoint.pt[0]
        blobs[i][2] = keypoint.size / 2

    return blobs

def tiled(image):

    blobs = detect(image,
                   invert=True,
                   maxThreshold=200,
                   filterByArea=True,
                   minArea=40,
                   filterByCircularity=True,
                   minCircularity=0.8,
                   )

    return blobs