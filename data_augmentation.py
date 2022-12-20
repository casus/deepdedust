import os
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
from skimage import io
import math
from PIL import Image,ImageDraw
import cv2


def data_split(source_path:str, dest_path:str , train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, shuffle=True):
    '''
    This function shuffles and segments input data. It creates the corresponding train,test and validation directories in the given destination directory.
    :param source_path: path to the directory containing the data to be split.
    :param dest_path: path to the destination directory where split data will be saved.
    :param train_ratio: The proportion of images to be used as the training dataset.
    :param test_ratio: The proportion of images to be used as the test dataset.
    :param val_ratio: The proportion of images to be used as the validation dataset.
    :return: None
    '''

    if train_ratio + test_ratio + val_ratio !=1.0:
        print('The sum of the split ratios should be equal to 1.0')
        return None

    train_filenames = []
    test_filenames = []
    val_filenames = []

    filelist = os.listdir(source_path)
    filelist_length = len(filelist)

    if shuffle:
        random.shuffle(filelist)

    train_filenames, test_filenames, val_filenames = np.split(filelist, [int(filelist_length * train_ratio),
                                                                         int(filelist_length * (train_ratio + test_ratio))])
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    print('Files are being copied!')
    
    if train_ratio>0.0:
        count = 0
        train_path = os.path.join(dest_path, 'train')
        os.mkdir(train_path)
        for file in train_filenames:
            shutil.copy(os.path.join(source_path,file),os.path.join(train_path,file))
            count+=1
            

    if test_ratio > 0.0:
        test_path = os.path.join(dest_path, 'test')
        os.mkdir(test_path)
        for file in test_filenames:
            shutil.copy(os.path.join(source_path,file),os.path.join(test_path,file))

    if val_ratio>0.0:
        count = 0
        val_path = os.path.join(dest_path, 'val')
        os.mkdir(val_path)
        for file in val_filenames:
            shutil.copy(os.path.join(source_path,file),os.path.join(val_path,file))
            count+=1
            

    print('Train_size:', len(train_filenames))
    print('Test_size:', len(test_filenames))
    print('Validation_size:', len(val_filenames))





def reshape_images(path):
    count = 0
    for file in os.listdir(path):
        raw_image = io.imread(os.path.join(path,file))
        stacked_img = np.stack((raw_image,)*3, axis=-1)
        io.imsave(os.path.join(path + '_RGB',file),stacked_img)
        reshaped_image = io.imread(os.path.join(path,file))
        if count <2:
            print(reshaped_image.shape)
            count+=1

    print('Reshaped all images successfully')

def convert_0_to_255(path):
    '''
    This function applies min-max normalization to all images in a source folder.
    '''

    for file in os.listdir(path):
        gt_image = io.imread(os.path.join(path,file))
        min = gt_image.min()
        max = gt_image.max()
        gt_image = gt_image.astype('float32')
        # MinMax Normalization for better image contrast.
        gt_image -= min
        gt_image /= (max - min)
        gt_image *= 255
        io.imsave(os.path.join(path,file),gt_image)



def reshape_images_opencv(path):
    '''
    
    This function converts a 2D image to a stacked 3D image.
    :param path: Path to the source directory.
    '''
    
    for file in os.listdir(path):
        raw_image = io.imread(os.path.join(path,file))
        img2 = np.zeros_like(raw_image)
        img2[:, :, 0] = raw_image[:, :, 0]
        img2[:, :, 1] = raw_image[:, :, 1]
        img2[:, :, 2] = raw_image[:, :, 2]
        io.imsave(os.path.join(path + '3D',file),img2)
        reshaped_image = io.imread(os.path.join(path + '3D',file))
      

  





def flist_creator(path):
    '''

    :param path: Path to the parent directory. flists of its sub-folders will be created.
    :return: None
    '''

    if not os.path.isdir(path):
        print('Input should be a directory.')
        return None

    # get the list of directories
    dirs = os.listdir(path)
    os.mkdir(os.path.join(path, 'flists'))

    for dir_item in dirs:
        # modify to full path -> directory
        if dir_item is None:
            continue
        temp = dir_item
        dir_item = path + "/" + dir_item
        folder = os.listdir(dir_item)
        file_names = []
        for file in folder:
            if file is None:
                break
            file_name = dir_item + '/' + file
            file_names.append(file_name)
        filename = str(path + '/flists/' + temp + '.flist')
        # make output file if it does not exist.
        if not os.path.exists(filename):
            with open(filename, 'w') as fo:
                fo.write("\n".join(file_names))
                fo.close()
        print("Written file is: ", filename)

#flist_creator('D:/DeDustProject/data/final_images_zoomed/gt_images_zoomed_data1')

def create_random_rectangular_mask(dest_folder,no_of_masks):
    """
    
    This functions creates binary images of rectangular masks of desired area.
    :param dest_folder: Path to the destination directory.
    :param no_of_masks: Number of binary mask images to be generated.
    """
    count = 0
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    while count < no_of_masks:
        img = Image.new('RGB',(256,256))
        img = np.asarray(img,dtype='uint8')
        img_size = img.shape[1]
        #mask_size = np.random.randint(int(0.30 * img_size), int(0.40 * img_size)) # 10 - 20% mask area due to sqaured area measure
        mask_size = np.random.randint(int(0.56 * img_size),int(0.635 * img_size))  # 30 - 40% mask area due to sqaured area measure
        y1, x1 = np.random.randint(0, img_size - mask_size, 2)
        y2, x2 = y1 + mask_size, x1 + mask_size
            #masked_part = img[:, y1:y2, x1:x2]
        #masked_img = img.clone()
        img[y1:y2, x1:x2,:] = 255
        io.imsave(f'{dest_folder}/{count}.PNG',img)
        count+=1


#create_random_rectangular_mask('D:/DeDustProject/data/rectangular_masks/30_to_40',36112)


def generate_irregular_masks(H, W,NumOutputMasks):
    """

    :param H: height of output mask image
    :param W: width of output mask image
    :param NumOutputMasks: number of mask images to create
    :return: None
    """
    average_radius = math.sqrt(H*H+W*W) / 8

    min_num_vertex = 30   #Default value is 10
    max_num_vertex = 32  #Default value is 25
    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 10
    min_width = 60   #Defalut value is 15
    max_width = 62

    for im_number in range(NumOutputMasks):
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(1, 3)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex) #Random number of vertices.
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            #vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            w_vertices = np.arange(0,w,16)
            h_vertices = np.arange(0,h,16)
            vertex.append((np.random.choice(w_vertices,size=1), np.random.choice(h_vertices,size=1)))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius // 2),
                    0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex: #Draw ellipes of random thickness and angle connecting the vertices.
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        mask = np.asarray(mask, np.float32)
        mask = np.reshape(mask, (H, W))
        plt.imshow(mask, cmap='gray')
        plt.show()
        
def tiff_to_png(path_to_folder):
    '''
    
    This function converts tiff images to png format.
    :param path_to_folder: Path to the source directory.
    :return: None
    '''
    count = 0
    new_dir_path = path_to_folder + "_PNG"

    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

    for file in os.listdir(path_to_folder):
        raw_image = io.imread(os.path.join(path_to_folder,file))
        raw_image = np.asarray(raw_image, np.float32)
        io.imsave(os.path.join(new_dir_path,file[:-4]) + '.png', raw_image)
        count+=1
    

def rgbpng_to_8bitpng(path_to_folder):
    '''
    
    This function converts rgb images to 8-bit png format.
    :param path_to_folder: Path to the source directory.
    :return: None
    '''
    
    for file in os.listdir(path_to_folder):
        raw_image = io.imread(os.path.join(path_to_folder,file))[:,:,1]
        raw_image = np.asarray(raw_image, np.uint8)
        io.imsave(f'D:/DeDustProject/data/zoomed_masked_images_png/{file[:-4]}.PNG', raw_image)



def min_max_255(mod_image):
    '''
    
    This function performs min-max normalization on a given image.
    :param mod_image: Source image.
    :return: None
    '''
    min = mod_image.min()
    max = mod_image.max()
    mod_image = mod_image.astype('float32')
    # MinMax Normalization for better image contrast.
    mod_image -= min
    mod_image /= (max - min)
    mod_image *= 255
    return mod_image


def add_images(image1_path,image2_path):
     '''
    
     This function adds two given images with differnet channels to overlay masks on a source image.
    :param image1_path: Path to first image.
    :param image1_path: Path to second image.
    :return: None
    '''
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)
    image1 = min_max(image1)
    
    img2 = np.zeros((image2.shape[0],image2.shape[1],3),dtype='float32')
    img2[:, :, 0] = image2
    img2[:, :, 1] = image2
    img2[:, :, 2] = image2
    img2 = np.asarray(img2,dtype='float32')
    img2 = min_max(img2)
    
    mod_image = (image1 * (1 - img2)) + img2

    #MinMax Normalization for better image contrast.
    mod_image = min_max_255(mod_image)
   
    io.imsave('D:/DeDustProject/data/final_images_zoomed/gt_images_zoomed_255_3/input_image1.TIF',mod_image)




def add_png_images(path_to_folder1,path_to_folder2):
    '''
    
    This function adds two given images to overlay masks on a source image.
    :param image1_path: Path to first image.
    :param image1_path: Path to second image.
    :return: None
    '''
    folder1_filelist = os.listdir(path_to_folder1)
    folder2_filelist = os.listdir(path_to_folder2)
    for imagefile1,imagefile2 in zip(folder1_filelist,folder2_filelist):
        img1 = io.imread(os.path.join(path_to_folder1,imagefile1))
        img2 = io.imread(os.path.join(path_to_folder2, imagefile2))
        img2[img2>0] = 255
        img1 = np.asarray(img1,dtype='float32')
        img2 = np.asarray(img2, dtype='float32')
        img1 = min_max(img1)
        img2 = min_max(img2)
        added_image = (img1 * (1-img2)) + img2
        added_image = cv2.normalize(added_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        io.imsave(f'D:/DeDustProject/data/Artifact/{imagefile2[:-4]}.PNG',added_image)




