import cv2 as cv
import numpy as np
import MNIST_Classify as mnist

# Begin video capture
cap = cv.VideoCapture(0)

# Set delay to 1ms
delay = 0

# Keycodes
ESCAPE_KEY = 27
A_KEY = 97
D_KEY = 100
W_KEY = 119
S_KEY = 115

#Thresholding values
min_thresh = 100
max_thresh = 200

# Contour Lengths
min_contour = 50
max_contour = 200

# Nearest distance to centers of contours
nearest_point_thres = 150

# Check if webcam is opened correctly
if not cap.isOpened():
    raise IOError("Unable to open webcam")

# Processing that is done on frame
def processFrame(frame):
    frame_processed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_processed = cv.Canny(frame_processed, min_thresh, max_thresh)
    th, frame_processed = cv.threshold(frame_processed, 100, 255, cv.THRESH_BINARY)
    # Find contours within image
    contours, hierarchy = cv.findContours(frame_processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.putText(frame_processed, f'min: {min_thresh}, max: {max_thresh}', (10,20), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv.LINE_AA)
    return frame_processed, contours

# Get average of points to find centers of contours
def get_contour_center(contours):
    centers = []
    for contour in contours:
        x_total = 0
        y_total = 0
        # Reject contours outside our desired length
        if len(contour) > min_contour and len(contour) < max_contour:
            for point in contour:
                x_total += point[0][0]
                y_total += point[0][1]
 
            average_x = x_total/len(contour)
            average_y = y_total/len(contour)
            distanceToNearestPoint = 1000
            # Check distances to other points
            if len(centers) > 0:
                for point in centers:
                    distance = np.sqrt((average_x - point[0])**2 + (average_y - point[1])**2)
                    if(distance < distanceToNearestPoint):
                        distanceToNearestPoint = distance
            # Only append to centers list if another point isn't already in proximity
            if distanceToNearestPoint > nearest_point_thres or len(centers) == 0:
                centers.append([int(average_x), int(average_y)])
            

    return centers

# Process images for classification
def process_image(img):
    img = img/255.0
    th,img = cv.threshold(img, 0.5, 1, cv.THRESH_BINARY_INV)
    img = cv.resize(img,(20,20),interpolation=cv.INTER_AREA)
    img = cv.copyMakeBorder(img, 4,4,4,4, cv.BORDER_CONSTANT, None, [0,0,0])
    return img

# Classify images using tensorflow model
def classify_image(image):
    image = process_image(image)
    num = mnist.classify_single(image)
    #cv.imshow(f"Classified: {num}", image) # For troubleshooting
    return num

# Loop to capture frames
while True:
    # Read frame
    ret, frame = cap.read()

    # Resize frame
    frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)
    frame_processed, contours = processFrame(frame)
    # Copy clean frame that has no drawings on it in grayscale
    frame_cpy = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    images_to_classify = []
    centers_of_images = []

    # Draw graphics
    #cv.drawContours(frame, contours, -1, (0,255,0), 1) # For troubleshooting
    centers = get_contour_center(contours)

    # Ensure arrays are cleared
    images_to_classify.clear()
    centers_of_images.clear()

    for center in centers:
        # Get center x and y
        c_x = center[0]
        c_y = center[1]

        # Draw centers and borders
        #frame = cv.circle(frame, center, 1, (0, 0, 255), 2)
        frame = cv.rectangle(frame, (c_x+40, c_y+40), (c_x-40,c_y-40), (0,255,0), 1)

        # Grab regions of interest from frame
        roi = frame_cpy[(c_y-40):(c_y+40), (c_x-40):(c_x+40)]
        if roi.size > 0:
            images_to_classify.append(roi)
            centers_of_images.append([c_x, c_y])
    
    # Classify and display the predicted number of all ROI's
    for indx, image in enumerate(images_to_classify):
        number, prob = classify_image(image)
        x = centers_of_images[indx][0]
        y = centers_of_images[indx][1]
        cv.putText(frame, f'Num: {number} Prob: {prob:.4f}', (x-40,y-50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
        #print(number)

    # Show frame
    cv.imshow('Input', frame)
    cv.imshow('Processed', frame_processed)

    # Wait for "delay" ms before going to next frame
    key = cv.waitKey(delay) 
    # If Escape key is pressed exit loop
    if key == ESCAPE_KEY:
        break
    if key == D_KEY:
        min_thresh += 1
    if key == A_KEY:
        min_thresh -= 1
    if key == W_KEY:
        max_thresh += 1
    if key == S_KEY:
        max_thresh -= 1

cap.release()
cv.destroyAllWindows()