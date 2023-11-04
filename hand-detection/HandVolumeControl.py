import cv2
import time
import numpy as np
import HandTrackingSimModule as htm
import math
import osascript

###############################################
#wCam, hCam = 64, 64
###############################################

cap = cv2.VideoCapture(0)
baseTime = 0
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectConf=0.7)

minVol = 0
maxVol = 100

# result = osascript.osascript('get volume settings')
# print(result)
# osascript.osascript("set volume output volume {}".format(50))
# result = osascript.osascript('get volume settings')
# print(result)



while True:

    success, img = cap.read()
    # detect hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print tip of thumb and tip of pointer finger
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        cx1, cy1 = lmList[4][1], lmList[4][2]
        cx2, cy2 = lmList[8][1], lmList[8][2]
        clx, cly = (cx1 + cx2)//2, (cy1 + cy2)//2

        cv2.circle(img, (cx1, cy1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx2, cy2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (cx1, cy1), (cx2, cy2), (255, 0, 255), 3)
        cv2.circle(img, (clx, cly), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(cx2-cx1, cy2-cy1)
        # print(length)

        vol = np.interp(length, [50, 300], [minVol, maxVol])
        osascript.osascript("set volume output volume {}".format(vol))


    # frame rate stuff
    currTime = time.time()
    fps = 1 / (currTime - baseTime)
    baseTime = currTime

    # make image mirror
    imgFlip = cv2.flip(img, 1)

    # print text and show
    cv2.putText(imgFlip, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    cv2.imshow("Image", imgFlip)

    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break