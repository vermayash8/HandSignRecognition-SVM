import cv2
import constants as C
from imageOps import drawAndGetRoi


def gatherDataForGesture(gestureName):
    camera = cv2.VideoCapture(0)
    started = False
    counter = 0
    while True:
        ret, frame = camera.read()
        roi = drawAndGetRoi(frame)
        if started:
            # Display a message on video that recording is on.
            print '[VERBOSE] Capturing image ', counter
            # Save the roi to dataset.
            roi = cv2.resize(roi, (100, 100))
            cv2.imwrite(C.path_data_storage + C.dataset_folder_name + gestureName + str(counter) + ".png", roi)
            counter = counter + 1
            if counter >= 100:
                print '[INFO] %d images captured, breaking.', counter
                break
        # Show the frame.
        cv2.imshow("Input Window", frame)
        # Handle keypress events.
        keyPressed = cv2.waitKey(66) & 0xFF
        if keyPressed == ord('s'):
            started = True
        elif keyPressed == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    gatherDataForGesture("test")