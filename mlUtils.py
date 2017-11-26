import cv2
import constants as C
import imageOps
import os
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

MODEL_FILE_NAME = "models/svm_saved_346_test.pkl"
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
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    pred = model.predict(H.reshape(1, -1))[0]
    # print "Predicted to be ", pred, " ", GESTURE_LIST[pred]
    return GESTURE_LIST[pred]


# Testing algorithm
def testAccuracy():
    correctly_predicted = 0
    total = 0
    testFolder = C.path_data_storage + C.test_folder_name
    for fileName in os.listdir(testFolder):
        if not fileName.endswith(".png"):
            print "[ERROR]: Unsupported file found."
            break
        print "[INFO] Processing ", fileName,
        img = cv2.imread(testFolder + fileName)
        predicted = predictGesture(img)
        correctAnswer = None
        for signName in GESTURE_LIST:
            if fileName.startswith(signName):
                correctAnswer = signName
        print "\tPredicted=", predicted, "\tCorrect=", correctAnswer
        if predicted == correctAnswer:
            correctly_predicted = correctly_predicted + 1
        total = total + 1
    print "\n--------\nTotal checked: ", total
    print "Correct Answers: ", correctly_predicted
    print "Accuracy", (1.0 * correctly_predicted)/total
