import math
import os
import cv2
from data_augmentation import min_max_255
import numpy as np
from skimage.draw import bezier_curve
from skimage.morphology import skeletonize
from skimage import io
from skimage.filters import threshold_otsu
from skimage import io,transform as t,img_as_ubyte
from matplotlib import pyplot as plt


def single_artifact_generator(path_to_artifact_image):
    '''
    
    This function generates biologically inspired artifdacts from a given image containing artifacts.
    :param path_to_artifact_image : PAth to the source image.
    '''

    source_image = cv2.imread(path_to_artifact_image)
    imgray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(imgray)  # Otsu thresholding
    binary = imgray > threshold   # Masked image.Multiplied by 0.7 to produce better masks.
    binary = img_as_ubyte(binary)
    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(
        binary, 8, cv2.CV_32S)

    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(imgray.shape, dtype="uint8")
    
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    for i in range(1, numLabels):
        # extract the connected component statistics for the current
        # label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area > 2000:
           
            componentMask = (labels == i).astype("uint8") * 1
            mask = cv2.bitwise_or(mask, componentMask)
            artifact_image = mask[y:y+h,x:x+w]
            

            skeleton = skeletonize(artifact_image)
            
            nonzero = np.nonzero(skeleton)
            # Returns a tuple of (nonzero_row_index, nonzero_col_index)
            # That is (array([0, 0, 1, 1, 2]), array([0, 2, 1, 2, 0]))

            nonzero_row = nonzero[0]
            nonzero_col = nonzero[1]


            count = 0
            

            nonzero_row = list(dict.fromkeys(nonzero_row))
            nonzero_col = list(dict.fromkeys(nonzero_col))
            #Differentaiate along the x and y-axis of the artifact skeleton to compute contraol points for bezier curve fitting.
            if not len(nonzero_row) > len(nonzero_col):
                
                dx = np.diff(nonzero_row)
                dy = np.diff(nonzero_col[:len(nonzero_row)])

            else:
                dx = np.diff(nonzero_row[:len(nonzero_col)])
                dy = np.diff(nonzero_col)


            try:
                d = abs(dy/dx)
            except ZeroDivisionError:
                print('Divide by zero error!')
                pass

            if not len(np.where(d==np.amax(d))[0]) == 1:
                max_index = int(np.where(d == np.amax(d))[0][0])
            else :
                max_index = int(np.where(d == np.amax(d))[0])



            
            
            # Returns a tuple of (nonzero_row_index, nonzero_col_index)
            # That is (array([0, 0, 1, 1, 2]), array([0, 2, 1, 2, 0]))

            #nonzero = np.nonzero(mask)
            length_row = len(nonzero_row)
            length_col = len(nonzero_col)
            #print('length is ',length_row,length_col)
            x0 = nonzero_row[0]
            y0 = nonzero_col[0]
            x1 = nonzero_row[max_index]
            y1 = nonzero_col[max_index]
            

            x2 = nonzero_row[length_row -1]
            y2 = nonzero_col[length_col-1]

           

            rr,cc = bezier_curve(x0,y0,x1,y1,x2,y2,weight=3)
            img= np.zeros((2160,2160),dtype='uint8')
            img[rr,cc] = 255

            
            nonzero_img = np.nonzero(img)
            nonzero_row = nonzero_img[0]
            nonzero_col = nonzero_img[1]

            count = 0
            img1 = np.zeros((artifact_image.shape[0],artifact_image.shape[1]),dtype='uint8')

            for i, j in zip(nonzero_row, nonzero_col):
               
                count += 1
                #Introduce random thickness into the mimicked artifact.
                thickness_x = np.random.randint(0, 10)
                thickness_y = np.random.randint(0, 10)
                
                offset_x_1 = 0
                offset_x_2 = 0
                offset_y_1 = 0
                offset_y_2 = 0
                img1[i - thickness_x + offset_x_1:i + thickness_x + offset_x_2,j - thickness_y + offset_y_1:j + thickness_y + offset_y_2] = 255 #Final artifact

                #io.imsave('artificial_artifact.PNG',img1)

