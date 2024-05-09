import os
import shutil

import cv2
import numpy as np
import time
from time import time, strftime
import sys


# Align and stack images with ECC method
# Slower but more accurate
def stackImagesECC(file_list, save_intermediates=True):
    M = np.eye(3, 3, dtype=np.float32)

    first_image = None
    stacked_image = None

    for idx, file in enumerate(file_list):
        image = cv2.imread(file,1).astype(np.float32) / 255
        print(file)
        if first_image is None:
            # convert to gray scale floating point image
            first_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            stacked_image = image
            image2 = stacked_image
        else:
            # Estimate perspective transform
            s, M = cv2.findTransformECC(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), first_image, M, cv2.MOTION_HOMOGRAPHY)
            w, h, _ = image.shape
            # Align image to first image
            image = cv2.warpPerspective(image, M, (h, w))
            image2 = image
            stacked_image += image
        if save_intermediates:
            splitext = os.path.splitext(file)
            filename = splitext[0] + '_warped_' +str(idx + 1) + splitext[1] #  __name__ + '_' +
            # image2 /= 2  # len(file_list)
            image2 = (image2 * 255).astype(np.uint8)
            cv2.imwrite(filename, image2)

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image


# Align and stack images by matching ORB keypoints
# Faster but less accurate
def stackImagesKeypointMatching(file_list, save_intermediates=True):

    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for idx, file in enumerate(file_list):
        print(file)
        image = cv2.imread(file,1)
        imageF = image.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
            image2 = imageF
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            image2 = imageF
            stacked_image += imageF
        if save_intermediates:
            splitext = os.path.splitext(file)
            filename = splitext[0] + '_warped_' + str(idx + 1) + splitext[1]  # __name__ + '_' +
            # image2 /= 2  # len(file_list)
            image2 = (image2 * 255).astype(np.uint8)
            cv2.imwrite(filename, image2)

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    return stacked_image

# ===== MAIN =====
# Read all files in directory
import argparse


if __name__ == '__main__':
    if not sys.stdout.isatty():
        import PySimpleGUI as sg

        file_list = sg.popup_get_file('', file_types=(('imgs', '.jpg .png .bmp'),), multiple_files=True, no_window=True, keep_on_top=True)
        method = 'ECC' if sg.popup_yes_no('ECC? (Slower but more accurate)', keep_on_top=True, auto_close=True, auto_close_duration=5) == 'Yes' else 'ORB'
        # import time
        output_image = strftime("%Y-%m-%d %H.%M.%S") + '.jpg'
        show = True if sg.popup_yes_no('display img?', keep_on_top=True, auto_close=True, auto_close_duration=5) == 'Yes' else False
    else:
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('input_dir', help='Input directory of images ()')
        parser.add_argument('output_image', help='Output image name')
        parser.add_argument('--method', help='Stacking method ORB (faster) or ECC (more precise)')
        parser.add_argument('--show', help='Show result image',action='store_true')
        args = parser.parse_args()

        image_folder = args.input_dir
        if not os.path.exists(image_folder):
            print("ERROR {} not found!".format(image_folder))
            exit()

        file_list = os.listdir(image_folder)
        file_list = [os.path.join(image_folder, x)
                     for x in file_list if x.endswith(('.jpg', '.png','.bmp'))]

        if args.method is not None:
            method = str(args.method)
        else:
            method = 'KP'

        output_image = args.output_image
        show = args.show

    file_list2 = []
    import unidecode
    for idx, file in enumerate(file_list):
        if file != unidecode.unidecode(file):
            shutil.copy(file, unidecode.unidecode(file))
        file_list2.append(unidecode.unidecode(file))
    file_list = file_list2

    tic = time()

    if method == 'ECC':
        # Stack images using ECC method
        description = "Stacking images using ECC method"
        print(description)
        stacked_image = stackImagesECC(file_list)

    elif method == 'ORB':
        #Stack images using ORB keypoint method
        description = "Stacking images using ORB method"
        print(description)
        stacked_image = stackImagesKeypointMatching(file_list)

    else:
        print("ERROR: method {} not found!".format(method))
        exit()

    print("Stacked {0} in {1} seconds".format(len(file_list), (time()-tic) ))


    print("Saved {}".format(output_image))
    cv2.imwrite(str(output_image), stacked_image)

    if show:
        cv2.imshow(description, stacked_image)
        cv2.waitKey(0)
