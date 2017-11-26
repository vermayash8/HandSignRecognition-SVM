import cv2
import car_controller
import dataGatherer
import imageOps
import mlUtils

def main():
    print "\n== == == == Hand sign recognizer == == == == "
    print "1. Generate dataset.\n2. Train model from generated dataset.\n3. Make predictions."
    choice = input("Choice: ")
    if choice == 1:
        print "Chose 1"
    elif choice == 2:
        trainModel()
    elif choice == 3:
        makePredictions()
    else:
        print "Invalid choice"
    main()


def generateDataset():
    gestureList = mlUtils.GESTURE_LIST
    for signName in gestureList:
        print "[INFO] Gathering for ", signName
        dataGatherer.gatherDataForGesture(gestureList)


def trainModel():
    # These will be used when dealing with multiple versions of datasets and models.
    # print "Enter the datasetName: "
    # print "Enter modelName: "
    mlUtils.trainHog()


def makePredictions():
    camera = cv2.VideoCapture(0)
    gameControl = False
    predictionStarted = False
    while True:
        ret, frame = camera.read()
        roi, frame = imageOps.drawAndGetRoi(frame)
        if predictionStarted:
            roi = cv2.resize(roi, (100, 100))
            prediction = mlUtils.predictGesture(roi)
            cv2.putText(frame, prediction, (450, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), thickness= 3)
            if gameControl:
                car_controller.performOperation(prediction)
        else:
            cv2.putText(frame, "Press s to start predictions", (20, 400), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow("CapturingWindow", frame)

        keyPressed = cv2.waitKey(100) & 0XFF
        if keyPressed == ord('q'):
            break
        elif keyPressed == ord('s'):
            predictionStarted = True
        elif keyPressed == ord('c'):
            gameControl = True
    camera.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
