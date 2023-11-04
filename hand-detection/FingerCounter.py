import cv2
import time
import os
import HandTrackingSimModule as htm

cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

baseTime = 0
currTime = 0

detector = htm.handDetector()

while True:

    success, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)

    index = 8
    handForce = False

    if len(lmList) != 0:
        while index <= 20:
            # check location of both x and y fingers, and calculate if hand is up or down
            if (lmList[index][2] < lmList[index - 2][2] and lmList[index][2] < lmList[0][2]) or (
                    lmList[index][2] > lmList[index - 2][2] and lmList[index][2] > lmList[0][2]) or (
                    lmList[index][1] < lmList[index - 2][1] and lmList[index][1] < lmList[0][1]) or (
                    lmList[index][1] > lmList[index - 2][1] and lmList[index][1] > lmList[0][1]):
                handForce = True
            else:
                handForce = False
                break

            index += 4
    print(handForce)

    if handForce == True:
        centerXAv = (int(lmList[17][1] + lmList[13][1] + lmList[9][1] + lmList[5][1] + lmList[0][1]))/5
        centerYAv = (int(lmList[17][2] + lmList[13][2] + lmList[9][2] + lmList[5][2] + lmList[0][2]))/5
        cv2.circle(img, (int(centerXAv), int(centerYAv)), 10, (255, 0, 255), cv2.FILLED)


    imgFlip = cv2.flip(img, 1)
    currTime = time.time()
    fps = 1 / (currTime - baseTime)
    baseTime = currTime
    cv2.putText(imgFlip, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    cv2.imshow("Image", imgFlip)

    cv2.waitKey(1)
