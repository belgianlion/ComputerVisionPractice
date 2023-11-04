import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def averaged_slope_intercept(image, lines):
    left_fit_solid = []
    left_fit_dotted = []
    right_fit_solid = []
    right_fit_dotted = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 100:
                left_fit_dotted.append((slope,intercept))
            else:
                left_fit_solid.append((slope, intercept))
        else:
            if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 100:
                right_fit_dotted.append((slope,intercept))
            else:
                right_fit_solid.append((slope, intercept))
        left_fit_average = np.average(left_fit_solid, axis=0)
        right_fit_average = np.average(right_fit_solid, axis=0)
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, "\n")
        return None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 80)
    return canny

def display_lines(image, lines):
    #zeros_like function displays an image
    #with the same resolution as its perameters
    #but with all the pixel values = to 0
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #line.reshape() takes a 2d array and makes it 1d
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (225, 0, 0), 10)
    return line_image

def display_lines_RAW(image, lines):
    red = 255
    green = 0
    #zeros_like function displays an image
    #with the same resolution as its perameters
    #but with all the pixel values = to 0
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #line.reshape() takes a 2d array and makes it 1d
            x1, y1, x2, y2 = line.reshape(4)
            if (np.sqrt((x1-x2)**2 + (y1-y2)**2) < 110):
                red = 255
                green = 0
            else:
                red = 0
                green = 255
            cv2.line(line_image, (x1, y1), (x2, y2), (0, green, red), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=5, maxLineGap=20)
    averaged_lines = averaged_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    line_image_raw = display_lines_RAW(frame, lines)
    combo_image_line = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    combo_image = cv2.addWeighted(frame, 0.8, line_image_raw, 1, 1)
    combo_av_RAW = cv2.addWeighted(combo_image, 0.8, line_image, 1, 1)
    canny_combo = cv2.addWeighted(canny_image, 0.8, line_image, 1, 1)
    #lineDef = line_color_type(frame)
    cv2.imshow('result', combo_av_RAW)
    #cv2.imshow('line data', line_image_raw)
    cv2.imshow('edge data', canny_combo)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
