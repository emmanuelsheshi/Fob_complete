import cv2
import mediapipe as mp
import time
import serial

from pynput.keyboard import Key, Listener
import serial.tools.list_ports
def get_ports():
    ports = serial.tools.list_ports.comports()

    return ports


def findArduino(portsFound):
    commPort = 'None'
    numConnection = len(portsFound)

    for i in range(0, numConnection):
        port = foundPorts[i]
        strPort = str(port)

        if 'Arduino' in strPort:
            splitPort = strPort.split(' ')
            commPort = (splitPort[0])

    return commPort

foundPorts = get_ports()
connectPort = findArduino(foundPorts)



def  on_press(key):
    print("press")

def on_release(key):
    print("release")


#weapond detection
import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('keras_model.h5')
# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()



class FaceMesh:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

    def detectFaceMesh(self, img, draw=True):
        # img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRgb)
        faces = []
        x, y = 0, 0
        xN, yN = 0, 0

        if self.results.multi_face_landmarks:

            xm = self.results.multi_face_landmarks[0].landmark[0].x
            ym = self.results.multi_face_landmarks[0].landmark[0].y
            iwL, ihL, icL = img.shape
            xN, yN = int(xm * 640), int(ym * 480)

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec,
                                               self.drawSpec)

                face = []

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])

                faces.append(face)

        return img, faces, (xN, yN)


def main():
    arduino = serial.Serial(connectPort, baudrate=115200, timeout=15)

    xCp, yCp = 320, 240
    xFce, yFce = 0, 0
    xEr, yEr = 0, 0
    errorThresh = 2

    trackingState = 1
    pitchTrackingState = 1

    # build string to send to microcontroller
    builtString = ''
    yaw_value = 100
    tilt_value = 58

    # mode selection
    objectDetected = 0
    state = 1

    # cap = cv2.VideoCapture('videos/1.mp4')
    cap = cv2.VideoCapture(0)

    ctime, ptime = 0, 0
    faceMesh = FaceMesh()



    while True:
      
       
        success, img = cap.read()

        img, faces, cp_face = faceMesh.detectFaceMesh(img)
        xFce, yFce = cp_face[0], cp_face[1]
        cv2.circle(img, (cp_face[0], cp_face[1]), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, (xCp, yCp), radius=5, color=(255, 0, 0), thickness=-1)
        xEr = xCp - xFce
        yEr = yCp - yFce


        k = cv2.waitKey(10)

        if k & 0xFF == ord("r"):
            print("reset position \n")
            yaw_value, tilt_value = 58, 30
            
            builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
            arduino.write(builtString.encode())

        elif k & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

            

        # if readchar.readchar() == 'r':
        #     yaw_value, tilt_value = 58, 30
        #     print("position reset \n")
        #     builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
        #     arduino.write(builtString.encode())
        #     arduino.flush()



        if len(faces) != 0:
            pass
            if xEr >= errorThresh:
                if trackingState:
                   if yaw_value < 180:
                    yaw_value += 2


            elif xEr < errorThresh:
                if trackingState:
                    if yaw_value > 0:
                        yaw_value -= 2

            if yEr <= errorThresh:
                if pitchTrackingState:
                    if tilt_value <= 100:

                        tilt_value += 1
                        print("pitch down here \n")

            elif yEr > errorThresh:
                if pitchTrackingState:

                    tilt_value -= 1
                    print("pitch up here \n")
                    if tilt_value <= 25:
                        tilt_value = 25

                # dead band setting for the yaw axis

            if -180 <= xEr <= 180:
                trackingState = 0
                print("stop tracking x")
            else:
                trackingState = 1
                print("continue tracking x")

            if -150 <= yEr <= 150:
                pitchTrackingState = 0
            else:
                pitchTrackingState = 1

            builtString = f'{yaw_value}' + ',' + f'{tilt_value}' + '\n'
            arduino.write(builtString.encode())
            arduino.flush()
            # print(arduino.readline().decode())
            print(xEr,yEr)




        ctime = time.time()
        frameRate = str(int(1 / (ctime - ptime)))
        ptime = ctime

        # print(cp_face)

        cv2.putText(img, frameRate, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


