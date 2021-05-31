import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ME597_image.png')
# cv2.imshow("Original Image",img)
lane_img = np.copy(img)

src_pts = np.float32([
[145,650],[800, 650],[650,540],[245,540]
])
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(145,650),(750, 650),(800,540),(245,540)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255,255,255))
    img_masked = cv2.bitwise_and(image,mask) #
    return img_masked

def canny(image):
    imgGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image Warped",imgGray)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)  ##(image, kernel size, deviation)
    # cv2.imshow("Blur Image Warped",imgBlur)
    imgCanny_Blur= cv2.Canny(imgBlur,80,200) ##Applies gaussian Blur byitself
    return imgCanny_Blur

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0), 5)  #Draw line wrt image space of blue color
    return line_image

def average_slope(image,lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 :
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    left_fit_average = np.average(left_fit, axis = 0)       ## axis = 0 calculates average along the vertical columns
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)

    return left_line, right_line

# def make_coordinates(image,line_parameters):
#     slope,intercept = line_parameters
#     y1 = img.shape[0]
#     y2 = int(y1*(3/5))  ## Upper Limit of Lane markings to be seen
#
#     x1 = int((y1 - intercept)/slope)
#     x2 = int((y2-intercept)/slope)
#
#     return np.array([x1, y1, x2, y2])

def mask(cv_image):
    height,width = cv_image.shape[0:2]
    # Convert from RGB to HSV
    # hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    #
    # lower_black = np.array([20,0,0])
    # upper_black = np.array([60,20,80])
    #lower_yellow = np.array([0,50,50])
    #upper_yellow = np.array([30,255,190])
    #
    # # Threshold the HSV image to get only yellow colors
    # mask = cv2.inRange(hsv, lower_black, upper_black)

    # Calculate centroid of the blob of binary image using ImageMoments
    ret, thresh = cv2.threshold(cv_image,127,255,0)
    m = cv2.moments(thresh,False)
    try:
        cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
    except ZeroDivisionError:
        cy, cx = height/2, width/2
    print(cx,cy)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(cv_image,cv_image, mask= mask)

    # Draw the centroid in the resultut image
    cv2.circle(cv_image,(int(cx), int(cy)), 100,(0,0,255),-1)

    return(cv_image)

def width_calc(lines,img):
    Lhs = np.zeros((2, 2), dtype = np.float32)
    Rhs = np.zeros((2, 1), dtype = np.float32)
    x_max = 0
    x_min = 2555
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Find the norm (the distances between the two points)
            normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32) # question about this implementation
            normal = normal / np.linalg.norm(normal)

            pt = np.array([[x1], [y1]], dtype = np.float32)

            outer = np.matmul(normal, normal.T)

            Lhs += outer
            Rhs += np.matmul(outer, pt) #use matmul for matrix multiply and not dot product

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness = 1)

            x_iter_max = max(x1, x2)
            x_iter_min = min(x1, x2)
            x_max = max(x_max, x_iter_max)
            x_min = min(x_min, x_iter_min)

    width = x_max - x_min
    print('width : ', width)
    # Calculate Vanishing Point
    vp = np.matmul(np.linalg.inv(Lhs), Rhs)
    print('vp is : ', vp)
    plt.plot(vp[0], vp[1], 'c^')
    cv2.imshow('Vanishing Point visualization',img)


Canny_img = canny(lane_img)     ##Canny edge detection with pre-built GaussianBlur
interest = region_of_interest(Canny_img)    ##Crop region of interest from Canny

cv2.imshow("Canny",Canny_img)
#width_calc(lines,interest)

dst_pts = np.float32([[0, 0], [500, 0],
                       [500, 600],
                       [0, 600]])



# H is the homography matrix
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (500,600))
canny_warped = canny(warped)
mask_image = mask(canny_warped)
lines = cv2.HoughLinesP(canny_warped,2, np.pi/180, 100, np.array([]),minLineLength = 40, maxLineGap = 5)    ##Line detection from region of interest

# averaged_lines = average_slope(warped, lines)     ##average out slope of all calculated lines into one left and one right line
#
# line_image = display_lines(warped,averaged_lines)      ##Display the lines as blue color on a black screen
#
# overlap_image = cv2.addWeighted(warped,0.7,line_image,1, 1) ## overlap lines on lane image detection(last argument is gamma)

# cv2.imshow("overlap",overlap_image)
# cv2.imshow("warped",warped)
# cv2.imshow("canny_warped",canny_warped)
# cv2.imshow("canny_warped", mask_image)
cv2.waitKey()





# cap = cv2.VideoCapture('test2.mp4')
# while (cap.isOpened()):
#     _, frame = cap.read()
#     Canny_img = canny(frame)     ##Canny edge detection with pre-built GaussianBlur
#
#     interest = region_of_interest(Canny_img)    ##Crop region of interest from Canny
#
#     lines = cv2.HoughLinesP(interest,2, np.pi/180, 100, np.array([]),minLineLength = 40, maxLineGap = 5)    ##Line detection from region of interest
#
#     averaged_lines = average_slope(frame, lines)     ##average out slope of all calculated lines into one left and one right line
#
#     line_image = display_lines(frame,averaged_lines)      ##Display the lines as blue color on a black screen
#
#     overlap_image = cv2.addWeighted(frame,0.7,line_image,1, 1) ## overlap lines on lane image detection(last argument is gamma)
#
#     cv2.imshow("result",overlap_image)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
