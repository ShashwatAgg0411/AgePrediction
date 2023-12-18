import cv2 as cv


# ______MAIN______

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
               "(38-43)", "(48-53)", "(60-100)"]


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = r"D:/OpenCv/deploy.prototxt.txt"
weightsPath = r"D:/OpenCv/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = r"D:/OpenCv/age_deploy.prototxt"
weightsPath = r"D:/OpenCv/age_net.caffemodel"
ageNet = cv.dnn.readNet(prototxtPath, weightsPath)

# InputPath="/path/to/your/input/image"
InputPath="shashwat1.jpg"
image = cv.imread(InputPath)

(h, w) = image.shape[:2]
blob = cv.dnn.blobFromImage(image, 1.0, (300, 300),
                            (104.0, 177.0, 123.0))


#detect face in image
print("[INFO] computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()
print(detections.shape)

for i in range(0, detections.shape[2]):
    # extract the confidence associated with the prediction
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        strtw = int(detections[0, 0, i, 3]*w)
        strth = int(detections[0, 0, i, 4]*h)
        endw = int(detections[0, 0, i, 5]*w)
        endh = int(detections[0, 0, i, 6]*h)

        face = image[strth:endh, strtw:endw]

        #predict age
        faceBlob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        ageNet.setInput(faceBlob)
        preds = ageNet.forward()

        #result
        i = preds[0].argmax()
        age = AGE_BUCKETS[i]
        ageConfidence = preds[0][i]

        #display
        text = "{}: {:.2f}%".format(age, ageConfidence * 100)
        print("[INFO] {}".format(text))
        y = strth - 10 if strth - 10 > 10 else strth + 10
        cv.rectangle(image, (strtw, strth), (endw, endh), (0, 0, 255), 2)
        cv.putText(image, text, (strtw, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv.imshow("Image", image)
cv.waitKey(0)
