import cv2 as cv
import numpy as np


def detect_face(frame, facenet):
    params_for_face_detections = {
        'scalefactor': 1.0,
        'size': (300, 300),
        'mean': (104.0, 177.0, 123.0),
        # 'mean': (78.4263377603, 87.7689143744, 114.895847746),
        'swapRB': False
    }
    blob = cv.dnn.blobFromImage(frame, **params_for_face_detections)
    facenet.setInput(blob)
    detections = facenet.forward()

    return detections


def extract_face(height, width, detections, detection):

    # variable
    START_WIDTH_INDEX = 3
    START_HEIGHT_INDEX = 4
    END_WIDTH_INDEX = 5
    END_HEIGHT_INDEX = 6

    box_start_width = int(detections[0, 0, detection, START_WIDTH_INDEX]*width)
    box_start_height = int(
        detections[0, 0, detection, START_HEIGHT_INDEX]*height)
    box_end_width = int(detections[0, 0, detection, END_WIDTH_INDEX]*width)
    box_end_height = int(detections[0, 0, detection, END_HEIGHT_INDEX]*height)

    bbox = [box_start_width, box_start_height, box_end_width, box_end_height]
    detectedface = frame[box_start_height:box_end_height,
                         box_start_width:box_end_width]

    return detectedface, bbox


def predict_age(agenet, AGE_BUCKETS, detectedface):
    params_for_age_prediction = {
        'scalefactor': 1.0,
        'size': (227, 227),
        'mean': (78.4263377603, 87.7689143744, 114.895847746),
        # 'mean':(104.0, 177.0, 123.0),
        'swapRB': False
    }

    blob = cv.dnn.blobFromImage(
        detectedface, **params_for_age_prediction)
    agenet.setInput(blob)
    predictions = agenet.forward()

    age_bucket_index = predictions[0].argmax()
    age_predicted = AGE_BUCKETS[age_bucket_index]
    pred_confidence = (predictions[0][age_bucket_index])
    prediction_confidence_percentage = pred_confidence*100
    prediction = "{},{:.2f}%".format(
        age_predicted, prediction_confidence_percentage)

    return prediction


def display_predictions(frame, bbox, text):
    box_start_width, box_start_height, box_end_width, box_end_height = bbox
    params_for_rectangle = {
        'pt1': (box_start_width, box_start_height),
        'pt2': (box_end_width, box_end_height),
        'color': (0, 255, 0),
        'thickness': 2
    }
    cv.rectangle(frame, **params_for_rectangle)
    text_start_height = box_start_height - \
        10 if box_start_height-10 > 10 else box_start_height+10

    params_for_text = {
        'org': (box_start_width, text_start_height),
        'fontFace': cv.FONT_HERSHEY_COMPLEX,
        'fontScale': 0.5,
        'color': (0, 255, 0),
        'thickness': 2,
    }
    cv.putText(frame, text, **params_for_text)

    return


def Predict(frame, facenet, agenet, MIN_CONFIDENCE_VALUE=0.5):
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                   "(38-43)", "(48-53)", "(60-100)"]
    (frame_height, frame_width) = frame.shape[:2]

    # Detecting face from the frame
    detections = detect_face(frame, facenet)

    # variables
    CONFIDENCE_INDEX = 2

    # looping over detections
    for detection in range(detections.shape[2]):

        # finding confidence of a particular detection
        confidence = detections[0, 0, detection, CONFIDENCE_INDEX]
        # print(confidence)

        if confidence > MIN_CONFIDENCE_VALUE:

            # extracting face from the frame
            detectedface, boundary_box = extract_face(
                frame_height, frame_width, detections, detection)

            # variables
            detected_face_height = detectedface.shape[0]
            detected_face_width = detectedface.shape[1]
            minimum_threshold = 20

            if detected_face_height < minimum_threshold or detected_face_width < minimum_threshold:
                continue

            # predicting age
            prediction = predict_age(agenet, AGE_BUCKETS, detectedface)

            # display the predictions on the frame
            display_predictions(frame, boundary_box, prediction)

    return frame


# _____MAIN______

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = r"deploy.prototxt.txt"
weightsPath = r"D:/OpenCv/res10_300x300_ssd_iter_140000.caffemodel"
facenet = cv.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = r"D:/OpenCv/age_deploy.prototxt"
weightsPath = r"D:/OpenCv/age_net.caffemodel"
agenet = cv.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] turning on camera...")
video = cv.VideoCapture(0)
# video=cv.VideoCapture(r"http://cdn.streamonweb.com/footprints/streamonweb_fp22_4/playlist.m3u8")
# video=cv.VideoCapture(r"http://cdn.streamonweb.com/footprints/streamonweb_fp22_4/playlist.m3u8")

if video.isOpened() is False:
    print("error")

while True:

    # traversing the captured video frame by frame
    _, frame = video.read()
    result = Predict(frame, facenet, agenet)

    cv.imshow('video', result)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv.destroyAllWindows()
