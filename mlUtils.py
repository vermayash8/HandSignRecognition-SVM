import cv2
import constants as C
import imageOps
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

MODEL_FILE_NAME = "models/svm_saved_346.pkl"
# List of all gestures that can be predicted. (Order matters)
GESTURE_LIST = ["Go", "Left", "Right", "Stop"]


# Training algorithm
def trainHog():
    data, labels = getHogDataLabels()
    print "****"
    print "[INFO] Encoding labels..."
    label_encoder = LabelEncoder().fit(GESTURE_LIST)
    labels = label_encoder.fit_transform(labels)
    print labels
    print "[INFO]: training classifier..."
    svcClassifier = SVC(kernel='linear', C=0.9, probability=True)
    svcClassifier.fit(data, labels)
    pickle.dump(svcClassifier, open(MODEL_FILE_NAME, 'wb'))
    print "[INFO] Done training. Model has been saved to ", MODEL_FILE_NAME


def getHogDataLabels():
    data = []
    labels = []
    folderName = C.path_data_storage + C.dataset_folder_name
    for fileName in os.listdir(folderName):
        if not fileName.endswith(".png"):
            print "[ERROR]: Unsupported file found."
            break
        print "[INFO] Processing ", fileName,
        img = cv2.imread(folderName + fileName)
        H, hoggedImage = imageOps.getHogVector(img)
        print H,
        label = None
        for signName in GESTURE_LIST:
            if fileName.startswith(signName):
                label = signName
        print label
        data.append(H.tolist())
        labels.append(label)
    return data, labels


# Prediction algorithm
def predictGesture(imageMat):
    H, hoggedImage = imageOps.getHogVector(imageMat)
    print "Predicted to be ",
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    pred = model.predict(H.reshape(1, -1))[0]
    print pred, " ", GESTURE_LIST[pred]
    return GESTURE_LIST[pred]


