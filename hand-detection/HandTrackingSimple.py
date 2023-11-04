import cv2
import mediapipe as mp
import time

# Run Web cam

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

baseTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                # Get location of each landmark in pixels
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                print(id, ":", cx, cy)
                if id == 0:
                    # Draws larger circle at given location based on ID input (this one is the lower palm)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    imgFlip = cv2.flip(img, 1)
    currTime = time.time()
    fps = 1 / (currTime - baseTime)
    baseTime = currTime



    # Adds text to image. Format is: (src, txt, loc, font, scale, color, thickness)
    cv2.putText(imgFlip, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    cv2.imshow("Image", imgFlip)

    # Converts NumLock to standard unicode as to prevent confusion.
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
