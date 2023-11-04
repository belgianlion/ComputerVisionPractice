import cv2
import mediapipe as mp
import time

# Run Web cam

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectConf=0.8, trackConf=0.8):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLandmark.landmark):
                    if draw == True:
                        self.mpDraw.draw_landmarks(img, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNum=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            currHand = self.results.multi_hand_landmarks[handNum]

            for id, lm in enumerate(currHand.landmark):
                # Get location of each landmark in pixels
                # print(id, lm)
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                #print(id, ":", cx, cy)

                lmList.append([id, cx, cy])

                if draw:
                    # Draws larger circle at given location based on ID input (this one is the lower palm)
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

        return lmList




def main():

    baseTime = 0
    currTime = 0

    detector = handDetector()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])

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

if __name__ == "__main__":
    main()