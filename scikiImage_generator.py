import random
import Augmentor
import cv2
import glob
import os
import numpy as np
from PIL import Image
from skimage import util
from skimage import io,transform as t,img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.morphology import opening,closing,disk
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt


def psnr(path_to_original_image,path_to_noisy_image):
    '''
    
    This function finds the peak signal to noise ratio for two given images.
    :param path_to_original_image : path to original noise-free image
    :param path_to_noisy_image : path to noisy image
    :returns Float value for the psnr of the two images.
    '''

    img1 = io.imread(path_to_original_image)
    img2 = io.imread(path_to_noisy_image)
    plt.imshow(img1,cmap='gray')
    plt.show()
    plt.imshow(img2, cmap='gray')
    plt.show()
    print(np.max(img1), np.max(img2))
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr

def one_psnr(path_to_image,axis=None, ddof=0):
    '''
     
    This function finds the peak signal to noise ratio for a single image.
    :param path_to_image : path to source image
    :returns Float value for the psnr of the given image.
    '''
    
    a = io.imread(path_to_image)
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    print(np.max(a))
    return np.where(sd == 0, 0, m/sd) * 100

path_to_original_img = "D:/DeDustProject/Artifacts/Artifacts/dapi_gfp_tritc_cy5/png/151130-AY-artifacts-10x-dapi-gfp-tritc-cy5_B02_s1_w1.png"
path_to_noisy_img = "D:/DeDustProject/Artifacts/Artifacts/cfp_6_exp_times/png/151130-AY-artifacts-10x-6cfp_B02_s1_w4.png"
#print(one_psnr(path_to_noisy_img))
print(psnr(path_to_original_img,path_to_noisy_img))

def simple_artifact_generator(image):

    '''
    
    This function generates images with simple square shaped masks placed randomly over the input images.
    :paramimage: Input image
    '''

    height,width = image.shape[0],image.shape[1]
    artifact = np.ones((200,200),dtype='uint16') * 255
    x0,y0 = np.random.randint(0,width - 200),np.random.randint(0,height - 200)
    image[x0:x0+200,y0:y0+200] = artifact

    return image

def artificial_artifacts_generator(gt_image_generator,masked_image_generator):

    '''
    
    This function generates images with irregular shaped artifacts.
    :param gt_image_generator : A ground truth image generator.
    :param masked_image_generator : A masked image generator.
    '''


    for gt_image, masked_image in zip(gt_image_generator, masked_image_generator):
        
        mod_image = cv2.add(gt_image,masked_image)
                     
        yield mod_image

def ground_truth_generator(path_to_ground_truth):

    '''
    
    This function generates ground truth images which do not contain any artifacts.
    :param path_to_ground_truth : Path to ground truth data.
    '''

    list_of_rows = ['E','F','G','H']
    list_of_columns = ['02','03','04','05']
    filelist = os.listdir(path_to_ground_truth)
    for file in filelist[:]:
        #Example : 151130-AY-artifacts-10x-6dapi_E01_s4_w3.tif
        if file.endswith(".TIF") and not file[-9:-4] =="Thumb" and file[30:33] !='E03' and file[30:33] != 'G04':
            if file[30] in list_of_rows and file[31:33] in list_of_columns:
                
                gt_image = io.imread(os.path.join(path_to_ground_truth, file), plugin='tifffile')
                min = gt_image.min()
                max = gt_image.max()
                gt_image = gt_image.astype('float32')
                #MinMax Normalization for better image contrast.
                gt_image -= min
                gt_image /= (max-min)
                
                yield gt_image


def convert_scale_0_to_255(path_to_ground_truth):
    '''
    
    This function converts an image from range [0,1] to range [0,255]
    :param path_to_ground_truth : Path to ground truth data.
    '''
    filelist = os.listdir(path_to_ground_truth)
    for file in filelist[:]:
        gt_image = io.imread(os.path.join(path_to_ground_truth, file), plugin='tifffile')
        min = gt_image.min()
        max = gt_image.max()
        gt_image = gt_image.astype('float32')
        # MinMax Normalization for better image contrast.
        gt_image -= min
        gt_image /= (max - min)
        gt_image *= 255
        io.imsave(os.path.join(path_to_ground_truth, file),gt_image)

def ground_truth_generator_0_to_255(path_to_ground_truth):
    '''
    
    This function generates ground truth images which do not contain any artifacts.
    :param path_to_ground_truth : Path to ground truth data.
    '''

    list_of_rows = ['E', 'F', 'G', 'H']
    list_of_columns = ['02', '03', '04', '05']
    filelist = os.listdir(path_to_ground_truth)
    for file in filelist[:]:
        # Example : 151130-AY-artifacts-10x-6dapi_E01_s4_w3.tif
        if file.endswith(".TIF") and not file[-9:-4] == "Thumb" and file[30:33] != 'E03' and file[30:33] != 'G04':
            if file[30] in list_of_rows and file[31:33] in list_of_columns:
                gt_image = io.imread(os.path.join(path_to_ground_truth, file), plugin='tifffile')
                min = gt_image.min()
                max = gt_image.max()
                gt_image = gt_image.astype('float32')
                # MinMax Normalization for better image contrast.
                gt_image -= min
                gt_image /= (max - min)
                gt_image *= 255
              
                yield file,gt_image


def masked_image_generator(path):

    '''
    
     This function implements Otsu thresholding followed by binary open & close operations to generate masks for pre-training.
     :param path : Path to the folder containing the images.
     '''

    filename = ''
    count = 0
    footprint = disk(1)
    list_of_rows = ['B', 'C', 'D','E']
    filelist = os.listdir(path)
    random.shuffle(filelist)
    for file in filelist[:]:
        #Example : 151130-AY-artifacts-10x-6cfp_A05_s1_w3.tif
        if file.endswith(".TIF") and not file[-9:-4] == "Thumb" and file[29] in list_of_rows :
            filename = file.split(".")[0]
            raw_image = io.imread(os.path.join(path,file), plugin='tifffile')
            threshold = threshold_otsu(raw_image) #Otsu thresholding

            if threshold > 200:

                binary = raw_image>threshold * 0.7 #Masked image.Multiplied by 0.7 to produce better masks.
                binary = img_as_ubyte(binary)
                binary_inverted = util.invert(binary) #Invert colorscale of image to make it suitable for morphological operations.
                closed = closing(binary_inverted, footprint) #Close operation on binary image.
                opened_and_closed = opening(closed,footprint) #Open operation after close operation to reproduce some fine-grained artifacts.
                opened_and_closed_inverted = util.invert(opened_and_closed) #Invert intensity scale of final image.(To get black background with white foreground )
                opened_and_closed_inverted =opened_and_closed_inverted.astype('float32')
                

                yield opened_and_closed_inverted

def zoom(img, zoom_factor=1.5):
     '''
     
     This function generates zoomed images.
     :param img : Image to be zoomed.
     :param zoom_factor = Zooming factor.
     '''

     zoomed_image = t.resize((t.rescale(img,zoom_factor,anti_aliasing=True)),(256,256),anti_aliasing=True) # Set size of resulting images to 256 * 256
     return zoomed_image

def zoomed_images_generator(image_generator):

    '''
    This function takes 16-bit raw images as input and generates their corresponding zoomed images using scikit-image.

    :param image_generator : Any image genertator.
    '''
    count = 0
    for raw_image in image_generator:
            cropping_factor = 12
            cropping_height, cropping_width = raw_image.shape[0]//cropping_factor,raw_image.shape[1]//cropping_factor
            height_offset = 0 # Used to vertically traverse a single image.
            while cropping_height + height_offset <= raw_image.shape[0]:
                width_offset = 0 # Used to horizontally traverse a single image.

                while cropping_width + width_offset <= raw_image.shape[1]:
                    cropped_image = raw_image[height_offset:cropping_height + height_offset,width_offset:cropping_width+width_offset]
                    # width_offset is used to horizontally traverse a single image.
                    zoomed_and_cropped_image = zoom(cropped_image)
                    zoomed_and_cropped_image = cv2.normalize(zoomed_and_cropped_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    mean = np.mean(zoomed_and_cropped_image)
                    if 50 > mean > 20 : #Min intensity criteria to avoid blank zoomed images.
                        
                        io.imsave(f'D:/DeDustProject/data/Pure_Artifacts/{count}.PNG',zoomed_and_cropped_image)
                        count+=1
                    if count == 8000 :
                        return
                    width_offset += 180 # raw_image.shape[0]/20 = 108. This gives 19 zoomed images.
                height_offset += cropping_height #Traverse down the image.

            #yield zoomed_and_cropped_image


def zoomed_images_generator_from_path(path):

    '''
    
    This function takes 16-bit raw images as input and generates their corresponding zoomed images using scikit-image.
    :param  image_generator : Any image genertator.
    '''

    for file in os.listdir(path):
        count = 0
        raw_image = io.imread(os.path.join(path,file),plugin='tifffile')
        cropping_factor = 12
        cropping_height, cropping_width = raw_image.shape[0] // cropping_factor, raw_image.shape[1] // cropping_factor
        height_offset = 0  # Used to vertically traverse a single image.
        while cropping_height + height_offset <= raw_image.shape[0]:
            width_offset = 0  # Used to horizontally traverse a single image.

            while cropping_width + width_offset <= raw_image.shape[1]:
                cropped_image = raw_image[height_offset:cropping_height + height_offset,
                                width_offset:cropping_width + width_offset]
                # width_offset is used to horizontally traverse a single image.
                zoomed_and_cropped_image = zoom(cropped_image)
                
                max = zoomed_and_cropped_image.max()
                mean  = zoomed_and_cropped_image.mean()
                #if max >= 0.140:
                if mean >= 14.0: #Min intensity criteria to avoid blank zoomed images.
                    io.imsave('D:/DeDustProject/data/zoomed_masked_images_new/{}_{}.PNG'.format(file[:-4],count),zoomed_and_cropped_image)
                    count += 1
                width_offset += 180  # raw_image.shape[0]/20 = 108. This gives 19 zoomed images.
            height_offset += cropping_height  # Traverse down the image.

        # yield zoomed_and_cropped_image


def zoomed_images_for_ground_truth(images_generator):

    '''
    
    This function takes 16-bit raw images as input and generates their corresponding zoomed images.

    :param images_generator:Any image generator.

    '''

    #path = 'D:/DeDustProject/Artifacts/Artifacts/cfp_6_exp_times'
    for filename,gt_image in images_generator:
            cropping_factor = 12
            cropping_height, cropping_width = gt_image.shape[0]//cropping_factor,gt_image.shape[1]//cropping_factor

            height_offset = 0 #Used to vertically traverse a single image.
            image_count = 0 #Used to keep track of zoomed images generated from a single ground truth image.
            while cropping_height + height_offset <= gt_image.shape[0]:
                width_offset = 0 #Used to horizontally traverse a single image.

                while cropping_width + width_offset <= gt_image.shape[1]:
                    cropped_image = gt_image[height_offset:cropping_height + height_offset,width_offset:cropping_width+width_offset]
                    # width_offset is used to horizontally traverse a single image.
                    zoomed_and_cropped_image = zoom(cropped_image,1.5)
                    max = zoomed_and_cropped_image.max()
                    mean = zoomed_and_cropped_image.mean()
                    mean_intensity_criteria = (filename[30] =='E' and mean >0.011*255) or (filename[30] =='F' and mean >0.011*255) or (filename[30] =='G' and mean >0.018*255) or (filename[30] =='H' and mean >0.018*255)
                    if max>=0.140*255:
                        if mean_intensity_criteria:
                            
                            io.imsave("D:/DeDustProject/data/final_images_zoomed/gt_images_zoomed_255/{}_{}_zoomed.TIF".format(filename[:-4],image_count), zoomed_and_cropped_image)
                            image_count += 1
                    width_offset += 180 # raw_image.shape[0]/12 = 180. This gives 11 zoomed images for each horizontal traversal of the image.
                height_offset += cropping_height #Traverse down the image.


def augment_images(path):

    '''
    
    This function generates augmented images from a collection of multiple source images.
    :param path : Path to source images.
    '''
    p = Augmentor.Pipeline(path)

    #Defining augmentation parameters and generating 36112 samples
    p.flip_left_right(0.5)
    p.black_and_white(0.1)
    p.rotate(0.3, 10, 10)
    p.skew(0.4, 0.5)     
    p.zoom(probability=0.2, min_factor=1.1, max_factor=1.3)
    p.sample(36112)

