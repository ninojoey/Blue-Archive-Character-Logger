import os
import cv2
import time
import random
import numpy as np
from pytesseract import pytesseract



PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEAR_X_OFFSET = 4
GEAR_Y_OFFSET = 6

BAD_COUNTER_MAX = 150000
UE_STAR_THRESHOLD = 0.03
OVERLAP_THRESHOLD = 0.35

# min miss:     0.612736701965332
# average miss: 0.72706709057093
# max hit:      0.1853787750005722
# average hit:  0.036196497934567
E_MATCH_THRESHOLD = 0.3

# min miss:     0.0605535731
# average miss: 0.069173962343484
# max hit:      0.0261527728
# average hit:  0.0080085342507
TIER_MATCH_THRESHOLD = 0.055

# max hit:     0.019280406
# average hit: 0.010435626252381
STAR_MATCH_THRESHOLD = 0.25
STAR_OVERLAP_THRESHOLD = 0.35
SCALE_INCREMENT = 0.01

# ccorr is bad because "we're dealing with discrete digital signals that have a
# defined maximum value (images), that means that a bright white patch of the image will basically
# always have the maximum correlation" and ccoeff pretty much fixes that problem.
# but for some reason ccoeff has been returning infinity in some cases... so yeah guess not that.
# sqdiff calculates the intensity in DIFFERENCE of the images. so you wanna look for the min if using sqdiff.
# also you want to use NORMED functions when template matching with templates of different sizes (ew're doing multiscale)
TEMPLATE_MATCH_METHOD = cv2.TM_SQDIFF_NORMED

# array of what min level you need to be to unlock the respective gear slot. slot1=lvl0, slot2=lvl15, slot3=lvl35
GEAR_SLOT_LEVEL_REQUIREMENTS = [0, 15, 35]


# TEMPLATE AND MASK IMAGES

STATS_TEMPLATE_IMAGE = cv2.imread("stats template.png", cv2.IMREAD_COLOR)
STATS_MASK_IMAGE = cv2.imread("stats mask.png", cv2.IMREAD_GRAYSCALE)

STATS_BOND_NAME_MASK_IMAGE = cv2.imread("stats bond name mask.png", cv2.IMREAD_GRAYSCALE)
BOND_MASK_IMAGE = cv2.imread("bond mask.png", cv2.IMREAD_GRAYSCALE)
NAME_MASK_IMAGE = cv2.imread("name mask.png", cv2.IMREAD_GRAYSCALE)
STATS_NAME_MASK_IMAGE = cv2.imread("stats name mask.png", cv2.IMREAD_GRAYSCALE)
STATS_BOND_MASK_IMAGE = cv2.imread("stats bond mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_MASK_IMAGE = cv2.imread("stats level mask.png", cv2.IMREAD_GRAYSCALE)

STATS_INFO_MASK_IMAGES = []
STATS_INFO_MASK_IMAGES.append(STATS_NAME_MASK_IMAGE)
STATS_INFO_MASK_IMAGES.append(STATS_BOND_MASK_IMAGE)
STATS_INFO_MASK_IMAGES.append(STATS_LEVEL_MASK_IMAGE)


STATS_STAR_MASK_IMAGE = cv2.imread("stats star mask.png", cv2.IMREAD_GRAYSCALE)
STAR_TEMPLATE_IMAGE = cv2.imread("star template.png", cv2.IMREAD_COLOR)
STAR_MASK_IMAGE = cv2.imread("star mask.png", cv2.IMREAD_GRAYSCALE)


EQUIPMENT_TEMPLATE_IMAGE = cv2.imread("equipment template.png", cv2.IMREAD_COLOR)
EQUIPMENT_MASK_IMAGE = cv2.imread("equipment mask.png", cv2.IMREAD_GRAYSCALE)



SKILLS_TEMPLATE_IMAGE = cv2.imread("skills template.png", cv2.IMREAD_COLOR)
SKILLS_1_STAR_MASK_IMAGE = cv2.imread("skills 1 star mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_2_STAR_MASK_IMAGE = cv2.imread("skills 2 star mask.png", cv2.IMREAD_GRAYSCALE)

SKILL_LEVEL_MASK_IMAGES = []
SKILL_1_LEVEL_MASK_IMAGE = cv2.imread("skill 1 level mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_2_LEVEL_MASK_IMAGE = cv2.imread("skill 2 level mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_3_LEVEL_MASK_IMAGE = cv2.imread("skill 3 level mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_4_LEVEL_MASK_IMAGE = cv2.imread("skill 4 level mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_1_LEVEL_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_2_LEVEL_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_3_LEVEL_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_4_LEVEL_MASK_IMAGE)

MAX_TEMPLATE_IMAGE = cv2.imread("max template.png", cv2.IMREAD_COLOR)
MAX_MASK_IMAGE = cv2.imread("max mask.png", cv2.IMREAD_GRAYSCALE)



UE_TEMPLATE_IMAGE = cv2.imread("ue template.png", cv2.IMREAD_COLOR)
UE_MASK_IMAGE = cv2.imread("ue mask.png", cv2.IMREAD_GRAYSCALE)
UE_STAR_MASK_IMAGE = cv2.imread("ue star mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_MASK_IMAGE = cv2.imread("ue level mask.png", cv2.IMREAD_GRAYSCALE)

STAR_UE_TEMPLATE_IMAGE = cv2.imread("star ue template.png", cv2.IMREAD_COLOR)
STAR_UE_MASK_IMAGE = cv2.imread("star ue mask.png", cv2.IMREAD_GRAYSCALE)



GEARS_TEMPLATE_IMAGE = cv2.imread("gears template.png", cv2.IMREAD_COLOR)
GEARS_MASK_IMAGE = cv2.imread("gears mask.png", cv2.IMREAD_GRAYSCALE)

GEAR_MASK_IMAGES = []
GEAR_1_MASK_IMAGE = cv2.imread("gear 1 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_2_MASK_IMAGE = cv2.imread("gear 2 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_3_MASK_IMAGE = cv2.imread("gear 3 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_MASK_IMAGES.append(GEAR_1_MASK_IMAGE)
GEAR_MASK_IMAGES.append(GEAR_2_MASK_IMAGE)
GEAR_MASK_IMAGES.append(GEAR_3_MASK_IMAGE)

GEAR_E_MASK_IMAGE = cv2.imread("gear E mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_TIER_MASK_IMAGE = cv2.imread("gear tier mask.png", cv2.IMREAD_GRAYSCALE)

E_TEMPLATE_IMAGE = cv2.imread("E template.png", cv2.IMREAD_COLOR)
E_MASK_IMAGE = cv2.imread("E mask.png", cv2.IMREAD_GRAYSCALE)


TIER_TEMPLATE_IMAGES = []
TIER_1_TEMPLATE_IMAGE = cv2.imread("tier 1 template.png", cv2.IMREAD_COLOR)
TIER_2_TEMPLATE_IMAGE = cv2.imread("tier 2 template.png", cv2.IMREAD_COLOR)
TIER_3_TEMPLATE_IMAGE = cv2.imread("tier 3 template.png", cv2.IMREAD_COLOR)
TIER_4_TEMPLATE_IMAGE = cv2.imread("tier 4 template.png", cv2.IMREAD_COLOR)
TIER_5_TEMPLATE_IMAGE = cv2.imread("tier 5 template.png", cv2.IMREAD_COLOR)
TIER_6_TEMPLATE_IMAGE = cv2.imread("tier 6 template.png", cv2.IMREAD_COLOR)
TIER_7_TEMPLATE_IMAGE = cv2.imread("tier 7 template.png", cv2.IMREAD_COLOR)
TIER_TEMPLATE_IMAGES.append(TIER_1_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_2_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_3_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_4_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_5_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_6_TEMPLATE_IMAGE)
TIER_TEMPLATE_IMAGES.append(TIER_7_TEMPLATE_IMAGE)
TIER_MASK_IMAGE = cv2.imread("tier mask.png", cv2.IMREAD_GRAYSCALE)


# indices
# [2:] = starting with index 2
# [:2] up to index 2
# [1:2] items from index 1 to 2
# [::2] = every 2 items, in other words skip 1
# [::-1] = every item, but in backwards order
# [2::2] starting with index 2, then every other item afterwards
# [:8:2] ending at index 8, get every other element starting from the beg
# [1:8:2] starting at index 1 and ending at index 8, get every other item
# [:,:] all elements in all rows. comma is normal.


# THINGS LEARNED AND USED
# template matching
#    multi scale matching
#    multi object matching
#    scaled search
#    methods
# non-maximum suppression
# masks
# json
#    loading
#    reading
#    will export for the website
# error methods (mse)
# subtracting/adding images
# object oriented
# combining images
# software design
# image processing
#    thresholding
#    morphology
#    contours
#    flood fill

def convertImageToString(image):
    pytesseract.tesseract_cmd = PATH_TO_TESSERACT
    imageToString = pytesseract.image_to_string(image)
##    print("imageToString:", imageToString)
    return imageToString


# resize an image based on a second image's dimensions
def resizeImage(imageToResize, imageToMatch):
    # get the dimensions of our matching image
    imageWidth = imageToMatch.shape[1]
    imageHeight = imageToMatch.shape[0]
    
    # resize our target image to the other's dimensions
    resizedImage = cv2.resize(imageToResize, (imageWidth, imageHeight))
    
    # return the image
    return resizedImage

# resize an image using a scale
def scaleImage(imageToResize, scale):
    # get the dimensions of our matching image
    imageWidth = imageToResize.shape[1]
    imageHeight = imageToResize.shape[0]

    scaledWidth = int(imageWidth * scale)
    scaledHeight = int(imageHeight * scale)
    
    # resize our target image to the other's dimensions
    resizedImage = cv2.resize(imageToResize, (scaledWidth, scaledHeight))
    
    # return the image
    return resizedImage


# given an image and a mask, crop the source within the area of the mask
# intended to be used with masks that only have 1 area which is rectangular
def cropImageWithMask(sourceImage, maskImage):
    maskImage = resizeImage(maskImage, sourceImage)
    
    # get the area where the mask is white then split it into x and y coordinate arrays
    croppedAreaLocations = np.where(maskImage == 255)
    croppedAreaXCoordinates = croppedAreaLocations[1]
    croppedAreaYCoordiantes = croppedAreaLocations[0]
    
    # the min and max values will be the corners of the croppedArea
    x1 = np.min(croppedAreaXCoordinates)
    x2 = np.max(croppedAreaXCoordinates)
    y1 = np.min(croppedAreaYCoordiantes)
    y2 = np.max(croppedAreaYCoordiantes)

    # return the cropped sourceimage
    return sourceImage[y1:y2, x1:x2]


# create a mask based on the transparency of a given image
def createMaskFromTransparency(image):
    ## if the image doesnt have an alpha channel (only 3 channels) then no need for a mask

    # get shape of our image
    imageChannelCount = image.shape[2]
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    # if our image is only bgr (no alpha), then we don't need to make a mask
    if imageChannelCount == 3:
        return None
        print("full whooite")
        maskImage = np.ones((imageHeight, imageWidth), np.uint8) * 255
        return maskImage
    
    
    ## create an image with its values being the max. then we adjust the pixels according to the alpha channel
    # start out with a blank image
    maskImage = np.ones((imageHeight, imageWidth), np.uint8) * 255
    
    # turn the img's transparency array into a scalar array.
    # an image's transparency is on index 3, and the max value is 255 (for completely visible)
    imageAlphaChannel = image[:,:,3] 
    imageAlphaScalar = imageAlphaChannel / 255.0
    
    # apply the scalar to our black image
    maskImage[:,:] = imageAlphaScalar[:,:] * maskImage[:,:]
    
    # return the image
    return maskImage


# given two images, and optional offsets, combine two images together,
# blending in their transparencies and return it
def combineTransparentImages(bgImage, fgImage, xOffset = 0, yOffset = 0):
    # get dimensions of our subarea, which is the fgImage
    fgImageWidth = fgImage.shape[1]
    fgImageHeight = fgImage.shape[0]
    
    # determine the area we're working with. offset + dimensions
    x1 = xOffset
    x2 = x1 + fgImageWidth
    y1 = yOffset 
    y2 = y1 + fgImageHeight

    # get the alpha channel of image2 and turn it into a scalar. turns 0-255 into 0-1
    fgImageAlphaChannel = fgImage[:,:,3]
    fgImageAlphaScalar = fgImageAlphaChannel / 255.0
    
    # calculate what image1"s alpha should be. it is image2"s complement 
    bgImageAlphaScalar = 1.0 - fgImageAlphaScalar

    # make a copy to work with and return
    combinedImage = bgImage.copy()

    # apply the scalar to each color channel
    for colorChannel in range (0, 3):
        # grab the color channel and apply the scalar to said channel
        bgImageColorChannel = bgImage[y1:y2, x1:x2, colorChannel]  # color channel
        bgImageColor = bgImageAlphaScalar * bgImageColorChannel    # alpha applied to color channel
        
        fgImageColorChannel = fgImage[:, :, colorChannel]
        fgImageColor = fgImageAlphaScalar * fgImageColorChannel


        # combine the colors from both images together
        combinedImage[y1:y2, x1:x2, colorChannel] = bgImageColor + fgImageColor

    return combinedImage


# given an image, process it to remove noise and then return
# the image with only the level showing
def processBondImage(statsImage):
    # convert the image to gray
    grayStatsImage = cv2.cvtColor(statsImage, cv2.COLOR_BGR2GRAY)
    
    # image threshold, basically turns it black and white, highlighting important textures and features
    processedStatsImage = cv2.threshold(grayStatsImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    
    statsImageContours = cv2.findContours(processedStatsImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    statsImageContours = statsImageContours[0] if len(statsImageContours) == 2 else statsImageContours[1]
    
    contoursImage = np.ones((statsImage.shape[0], statsImage.shape[1]), np.uint8) * 255
    # draw(imageToDrawOn, listOfContours, contourToDraw (negative is all), color, lineType)
    cv2.drawContours(contoursImage, statsImageContours, -1, 0, cv2.FILLED)
    
    
    statsBondNameImage = cropImageWithMask(contoursImage, STATS_BOND_NAME_MASK_IMAGE)

    filledStatsBondNameImage = statsBondNameImage.copy()
    # floodfill our image to black out the background
    startPoint = (0, 0)         # start at the first pixel
    newColor = (0, 0, 0, 255)   # color to flood with, black in this case
    cv2.floodFill(filledStatsBondNameImage, None, startPoint, newColor)
    
    resizedBondMask = resizeImage(BOND_MASK_IMAGE, statsBondNameImage)
    bondImage = cv2.bitwise_and(filledStatsBondNameImage, resizedBondMask)



    resizedNameMask = resizeImage(NAME_MASK_IMAGE, statsBondNameImage)
    invertedStatsBondNameImage = np.invert(statsBondNameImage)
    nameImage = cv2.bitwise_and(invertedStatsBondNameImage, resizedNameMask)

    

    # morph open our image to remove general noise
    kernelSizes = [1, 2, 3]
    morphedBondNames = ["ero", "dil", "manop", "op", "mancl", "cl"]
    morphedBonds = []
    
    for kernelIndex in range(len(kernelSizes)):
        kernelSize = kernelSizes[kernelIndex]
        
        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        
        erosion = cv2.erode(bondImage, kernel)
        dilation = cv2.dilate(bondImage, kernel)
        manualOpen = cv2.dilate(erosion, kernel)
        opening = cv2.morphologyEx(bondImage, cv2.MORPH_OPEN, kernel)
        manualClose = cv2.erode(dilation, kernel)
        closing = cv2.morphologyEx(bondImage, cv2.MORPH_CLOSE, kernel)
        
        morphedBonds.append(erosion)
        morphedBonds.append(dilation)
        morphedBonds.append(manualOpen)
        morphedBonds.append(opening)
        morphedBonds.append(manualClose)
        morphedBonds.append(closing)
    
    bond = 0
    name = "yes"

    for index in range(len(morphedBonds)):
        morphedBond = morphedBonds[index]
        bondNameImage = cv2.bitwise_or(morphedBond, nameImage)

        bondNameString = convertImageToString(bondNameImage)
        [bond, name] = bondNameString.split(" ", 1)
        
        cv2.imshow(str(time.time()), bondNameImage)

        print("kernel:", str(index % 3 + 1), morphedBondNames[index % 6])
        print("bond:", bond)
        print("name:", name)
    
    return bond, name



# given an image, process it to remove noise and then return
# the image with only the level showing
def processImage(colorImage, maskImage = None):
    # convert the image to gray
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
##    cv2.imshow("grayImage", grayImage)

    
    # image threshold, basically turns it black and white,
    # highlighting important textures and features
    processedImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
##    cv2.imshow("thresh", processedImage)

    



    countourMethods = [cv2.RETR_LIST, cv2.RETR_EXTERNAL, cv2.RETR_TREE, cv2.RETR_CCOMP]
    
    
    
    # find contours and remove small noise
    processedImage1 = processedImage.copy()
    processedImage2 = processedImage.copy()
    processedImage3 = processedImage.copy()
    processedImage4 = processedImage.copy()

    
##    for contour in contours:
##        area = cv2.contourArea(contour)
##        if area < 15:
##            cv2.drawContours(processedImage, [contour], -1, 0, 1)

    
    contours = cv2.findContours(processedImage1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(processedImage1, contours, -1, 0, 1)
    blankImage1 = np.ones((processedImage1.shape[0], processedImage1.shape[1]), np.uint8) * 255
    cv2.drawContours(blankImage1, contours, -1, 0, cv2.FILLED)
##    cv2.imshow("RETR_LIST", processedImage1)
    cv2.imshow("b RETR_LIST", blankImage1)

    
    contours = cv2.findContours(processedImage2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(processedImage2, contours, -1, 0, 1)
    blankImage2 = np.ones((processedImage1.shape[0], processedImage1.shape[1]), np.uint8) * 255
    cv2.drawContours(blankImage2, contours, -1, 0, cv2.FILLED)
##    cv2.imshow("RETR_EXTERNAL", processedImage2)
    cv2.imshow("b RETR_EXTERNAL", blankImage2)


    contours = cv2.findContours(processedImage3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(processedImage3, contours, -1, 0, 1)
    blankImage3 = np.ones((processedImage1.shape[0], processedImage1.shape[1]), np.uint8) * 255
    cv2.drawContours(blankImage3, contours, -1, 0, cv2.FILLED)
##    cv2.imshow("RETR_TREE", processedImage3)
    cv2.imshow("b RETR_TREE", blankImage3)


    contours = cv2.findContours(processedImage4, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(processedImage4, contours, -1, 0, 1)
    blankImage4 = np.ones((processedImage1.shape[0], processedImage1.shape[1]), np.uint8) * 255
    cv2.drawContours(blankImage4, contours, -1, 0, cv2.FILLED)
##    cv2.imshow("RETR_CCOMP", processedImage4)
    cv2.imshow("b RETR_CCOMP", blankImage4)
    

##    # morph open our image to remove general noise
##    kernel = np.ones((1, 1),np.uint8)
##    kernel2 = np.ones((1, 1),np.uint8)
##    
##    erosion = cv2.erode(processedImage1, kernel, iterations = 1)
##    cv2.imshow("erode", erosion)
##    
##    dilation = cv2.dilate(processedImage1, kernel,iterations = 1)
##    cv2.imshow("dilation", dilation)
##
##    manualOpen = cv2.dilate(erosion, kernel, iterations = 1)
##    cv2.imshow("manual open", manualOpen)
##
##    opening = cv2.morphologyEx(processedImage1, cv2.MORPH_OPEN, kernel)
##    cv2.imshow("opening", opening)
##
##    manualClose = cv2.erode(dilation, kernel, iterations = 1)
##    cv2.imshow("manual close", manualClose)
##
##    closing = cv2.morphologyEx(processedImage1, cv2.MORPH_CLOSE, kernel)
##    cv2.imshow("closing", closing)
    
##    processedImage1 = cv2.morphologyEx(processedImage1, cv2.MORPH_OPEN, kernel, iterations = 1)
##    processedImage3 = cv2.morphologyEx(processedImage3, cv2.MORPH_OPEN, kernel, iterations = 1)
##    processedImage4 = cv2.morphologyEx(processedImage4, cv2.MORPH_OPEN, kernel, iterations = 1)
##    
##    cv2.imshow("MORPH_OPEN1", processedImage1)
##    cv2.imshow("MORPH_OPEN2", processedImage2)
##    cv2.imshow("MORPH_OPEN3", processedImage3)
##    cv2.imshow("MORPH_OPEN4", processedImage4)



    # floodfill our image to black out the background
    startPoint = (10, 10)         # start at the first pixel
    newColor = (0, 0, 0, 255)   # color to flood with, black in this case
    cv2.floodFill(processedImage1, None, startPoint, newColor)
    cv2.floodFill(processedImage2, None, startPoint, newColor)
    cv2.floodFill(processedImage3, None, startPoint, newColor)
    cv2.floodFill(processedImage4, None, startPoint, newColor)
    
##    cv2.imshow("floodFill1", processedImage1)
##    cv2.imshow("floodFill2", processedImage2)
##    cv2.imshow("floodFill3", processedImage3)
##    cv2.imshow("floodFill4", processedImage4)



    
    # apply mask if there is one 
    if maskImage is not None:
        maskImage = resizeImage(maskImage, processedImage)
        processedImage1 = cv2.bitwise_and(processedImage1, maskImage)
        processedImage2 = cv2.bitwise_and(processedImage2, maskImage)
        processedImage3 = cv2.bitwise_and(processedImage3, maskImage)
        processedImage4 = cv2.bitwise_and(processedImage4, maskImage)
    blankImage1 = cropImageWithMask(blankImage1, STATS_BOND_NAME_MASK_IMAGE)
    blankImage2 = cropImageWithMask(blankImage2, STATS_BOND_NAME_MASK_IMAGE)
    blankImage3 = cropImageWithMask(blankImage3, STATS_BOND_NAME_MASK_IMAGE)
    blankImage4 = cropImageWithMask(blankImage4, STATS_BOND_NAME_MASK_IMAGE)
        #cv2.imshow("6", processedImage)


    

    processedImage1 = cv2.cvtColor(processedImage1, cv2.COLOR_GRAY2BGR)
    processedImage2 = cv2.cvtColor(processedImage2, cv2.COLOR_GRAY2BGR)
    processedImage3 = cv2.cvtColor(processedImage3, cv2.COLOR_GRAY2BGR)
    processedImage4 = cv2.cvtColor(processedImage4, cv2.COLOR_GRAY2BGR)
    
    # floodfill our image to black out the background
    startPoint = (0, 0)         # start at the first pixel
    newColor = (0, 0, 0, 255)   # color to flood with, black in this case
    cv2.floodFill(blankImage1, None, startPoint, newColor)
    cv2.floodFill(blankImage2, None, startPoint, newColor)
    cv2.floodFill(blankImage3, None, startPoint, newColor)
    cv2.floodFill(blankImage4, None, startPoint, newColor)

    cv2.imshow("masked1", blankImage1)
    cv2.imshow("masked2", blankImage2)
    cv2.imshow("masked3", blankImage3)
    cv2.imshow("masked4", blankImage4)
    
    print(convertImageToString(blankImage1))
    print(convertImageToString(blankImage2))
    print(convertImageToString(blankImage3))
    print(convertImageToString(blankImage4))
##    
    return processedImage1, processedImage2, processedImage3, processedImage4


# calculate and return the mean squared error between two given images, with an optional mask
def mse(colorImage1, colorImage2, maskImage = None):
    if maskImage is None:
        maskImage = createMaskFromTransparency(colorImage2)

    grayImage1 = cv2.cvtColor(colorImage1, cv2.COLOR_BGR2GRAY)
    grayImage2 = cv2.cvtColor(colorImage2, cv2.COLOR_BGR2GRAY)
    
    # store our image"s width and height. they should be the same
    imageWidth = grayImage1.shape[1]
    imageHeight = grayImage1.shape[0]

    maskedImage1 = cv2.bitwise_and(grayImage1, grayImage1, mask = maskImage)

    # subtract our image. will return a 2d array of 0"s and 1"s
    # 0 should represent if the pixels are the same. 1 if not
    imageDifference = cv2.subtract(maskedImage1, grayImage2)

    # sum up all the diff squared
    differenceSquared = np.sum(imageDifference ** 2)

    # calculate the mse
    imageArea = float(sourceImageHeight * sourceImageWidth)
    error = differenceSquared/imageArea
    
    return error, imageDifference


def returnErrAndDiff(template_img, subimage):
    # make sure both images are the same dimensions
    template_resized = cv2.resize(template_img, (subimage.shape[1], subimage.shape[0]))
    template_scaledGrayTemplateImage = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

    subimage_gray = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
    
    error, diff = mse(template_scaledGrayTemplateImage, subimage_gray)
    print("Image matching Error between the two images:", error)

    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(diff)
    print (text[:-1])
    
    return diff


# from an array of results, return results that are above a certain threshold.
# filter out results that have major overlap to avoid repeated hits
def nms(matchResults, matchLocations, matchWidth, matchHeight, overlapThreshold):
    # unpack our arrays
    x1MatchCoordinates = matchLocations[1]
    x2MatchCoordinates = x1MatchCoordinates + matchWidth
    y1MatchCoordinates = matchLocations[0]
    y2MatchCoordinates = y1MatchCoordinates + matchHeight
    
    # determine which order to run through the results/coordinates.
    # we do it based on the results, starting with the highest match.
    # in TM method sqdiff, the lowest value is the highest match. while the other
    # methods' highest value is the highest match.
    # but since np.argsort sorts from low to high, we reverse it with [::-1] if needed.
    # we create a dedicated array instead of just sorting the results array itself
    # since we're working with other arrays (the coordinates array and results array)
    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        indexOrder = np.argsort(matchResults)[::1]
    else:
        indexOrder = np.argsort(matchResults)[::-1]
    
    # get the area of the box. they should all be the same area
    matchArea = matchWidth * matchHeight

    # create an array to store our overlap percentages
    overlap = []

    # array to store the coordinates of our best matches, filtering out overlappers
    bestMatchLocations = []
    bestMatches = []
    

    # go through all our boxes starting with highest match to lowest match.
    # grab the coordinates of that box and store it into our filtered list.
    # go through the rest of the coordinates to see if any of them have overlap.
    # if any have overlap over the threshhold, we delete those values from our list
    while len(indexOrder) > 0:
        # get the index to work with. should be our current highest match result
        # which should be the first item in the list.
        bestMatchIndex = indexOrder[0]

        # get the location of the current best match and add it to our bestMatchLocations list
        x1BestMatchCoordinate = x1MatchCoordinates[bestMatchIndex]
        x2BestMatchCoordinate = x2MatchCoordinates[bestMatchIndex]
        y1BestMatchCoordinate = y1MatchCoordinates[bestMatchIndex]
        y2BestMatchCoordinate = y2MatchCoordinates[bestMatchIndex]
        
        bestMatchLocations.append((x1BestMatchCoordinate, y1BestMatchCoordinate))
        bestMatches.append(matchResults[bestMatchIndex])

        ## determine the overlap of the other matches with the current match.
        ## to find the overlapping area, the x and y values should be furthest away
        ## from the edges of the original. if comparing our x1 (top left) coordinate,
        ## we look to see if the other x1 coordinates are further away from the top left.
        # go through respective coordinates (x1, x2, y1, y2) and compare the
        # best match coordinate with its respective matchCoordinates array.
        x1OverlapCoordinates = np.maximum(x1BestMatchCoordinate, x1MatchCoordinates)
        x2OverlapCoordinates = np.minimum(x2BestMatchCoordinate, x2MatchCoordinates)
        y1OverlapCoordinates = np.maximum(y1BestMatchCoordinate, y1MatchCoordinates)
        y2OverlapCoordinates = np.minimum(y2BestMatchCoordinate, y2MatchCoordinates)
        
        ## calculate the area of overlap
        # we do a max of 0, because if you have a negative edges, that means there isn't 
        # any overlap and actually represents the area of empty space between the two matches.
        # and if you have 2 negative edges, the area will
        # still be positive, resulting in a false positive.
        overlapWidths  = np.maximum(0, x2OverlapCoordinates - x1OverlapCoordinates)
        overlapHeights = np.maximum(0, y2OverlapCoordinates - y1OverlapCoordinates)
        overlapAreas = overlapWidths * overlapHeights

        # calculate the percentage of overlap with our matched area
        overlapPercentages = overlapAreas / matchArea

        # delete the indices of matches where the overlap is over a certain threshold.
        # in this case, we delete the entries of the indicies so we don't
        # iterate over them later. also delete the one we just worked on cause it's done
        indexDelete = np.where(overlapPercentages > overlapThreshold)[0]
        if bestMatchIndex not in indexDelete:
            indexDelete = np.append(indexDelete, bestMatchIndex)
        
        indexOrder = np.setdiff1d(indexOrder, indexDelete, True)
        

    # return the coordinates of the filtered boxes
    return bestMatchLocations, bestMatches


#
def runAllTemplateMatchingMethods(graySourceImage, grayTemplateImage, maskImage):
    bestMatchVal = 0
    bestMatchResult = []
    
    for method in TEMPLATE_MATCH_METHODS_ARRAY:
        matchResult = cv2.matchTemplate(graySourceImage, grayTemplateImage, method, mask = maskImage)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(matchResult)
        

##        x1 = maxLoc[0]
##        x2 = x1 + grayTemplateImage.shape[1]
##        y1 = maxLoc[1]
##        y2 = y1 + grayTemplateImage.shape[0]
##        subimage = graySourceImage[y1:y2, x1:x2]
##        cv2.imshow(str(time.time()), subimage)
        
        

        if maxVal > bestMatchVal:
            bestMatchVal = maxVal
            bestMatchResult = matchResult
            bestMethod = method

    return bestMatchResult, str(bestMethod)
    


# template match given a scale. scale the template and template match
# used for when you know what the scale of the template will be in your source
def subimageScaledSearch(colorSourceImage, colorTemplateImage, maskImage = None, scale = 1.0):
    # make a grayscale copy of the main image
    graySourceImage = cv2.cvtColor(colorSourceImage, cv2.COLOR_BGR2GRAY)
    
    # make a grayscale copy of the template image, and store its width and height
    grayTemplateImage = cv2.cvtColor(colorTemplateImage, cv2.COLOR_BGR2GRAY)
    templateImageWidth = grayTemplateImage.shape[1]
    templateImageHeight = grayTemplateImage.shape[0]

    scaledWidth = int(templateImageWidth * scale)
    scaledHeight = int(templateImageHeight * scale)
    
    # resize our template and if needed, the mask 
    scaledGrayTemplateImage = cv2.resize(grayTemplateImage, (scaledWidth, scaledHeight))
    
    # get the results of our template match and extract the location of the best match from our results
    if maskImage is None:
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
    else:
        scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
##        matchResult, method = runAllTemplateMatchingMethods(graySourceImage, scaledGrayTemplateImage, scaledMaskImage)

    
    # extract our values from the template match result
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResult)

    bestResult = 0

    # our location differs by the method we're using
    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        bestResult = minVal
        x1 = minLoc[0]
        y1 = minLoc[1]
    else:
        bestResult = maxVal
        x1 = maxLoc[0]
        y1 = maxLoc[1]
    
    # compute the area of our subimage in our source image
    x2 = x1 + scaledWidth
    y2 = y1 + scaledHeight
    
    # get our subimage
    subimage = colorSourceImage[y1:y2, x1:x2]
    #print("best method:", method)

    # return the subimage along with some other values
    return subimage, scaledWidth, scaledHeight, bestResult, matchResult


# resize our template image at different scales to try to find the best match in our source image
# return the results of said match, and the location of best match
def subimageMultiScaleSearch (colorSourceImage, colorTemplateImage, maskImage = None):
    # make a grayscale copy of the main image, and store its width and height
    graySourceImage = cv2.cvtColor(colorSourceImage, cv2.COLOR_BGR2GRAY)
    sourceImageWidth = graySourceImage.shape[1]
    sourceImageHeight = graySourceImage.shape[0]
    
    # make a grayscale copy of the template image, and store its width and height
    grayTemplateImage = cv2.cvtColor(colorTemplateImage, cv2.COLOR_BGR2GRAY)
    templateImageWidth = grayTemplateImage.shape[1]
    templateImageHeight = grayTemplateImage.shape[0]
    
    
    # variables that keep track of values of our best match
    bestMatchLocation = (0, 0)
    bestWidth = templateImageWidth
    bestHeight = templateImageHeight
    bestScale = 1.0
    bestMatchResult = []
    bestMethod = ""
    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        bestMatch = float("inf")
    else:
        bestMatch = 0
    

    # variable to scale the template by
    scale = 1.0

    # values of our newly scaled template/image
    scaledWidth  = int(templateImageWidth * scale)
    scaledHeight = int(templateImageHeight * scale)

    # set our boundaries and a counter to increase efficiency
    xBoundary = sourceImageWidth
    yBoundary = sourceImageHeight
    badCounter = 0

    
    # scale up our template and match to look for the best match until
    # we hit one of original's dimensions or until we go on a 15 bad streak
    while scaledWidth <= xBoundary and scaledHeight <= yBoundary and badCounter < BAD_COUNTER_MAX:
        # resize our template after we"ve scaled it
        scaledGrayTemplateImage = cv2.resize(grayTemplateImage, (scaledWidth, scaledHeight))

        # find the location of our best match to the template in our resized image.
        # matchTemplate returns a 2d array of decimals. the decimal represents the match value
        # of the tempalte at that respective location on the image.
        if maskImage is None:
            matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
        else:
            scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
            matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
##            matchResult, method = runAllTemplateMatchingMethods(graySourceImage, scaledGrayTemplateImage, scaledMaskImage)

        # store our values into corresponding variables
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResult)

        # check to see if our current match is better than our best match
        if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
            if minVal < bestMatch:
                # if so, update our tracking variables and reset our bad counter
                bestMatch = minVal
                bestMatchLocation = minLoc
                bestWidth = scaledWidth
                bestHeight = scaledHeight
                bestScale = scale
                bestMatchResult = matchResult
                badCounter = 0
##                bestMethod = method
            else:
                badCounter += 1
        else:
            if maxVal > bestMatch:
                # if so, update our tracking variables and reset our bad counter
                bestMatch = maxVal
                bestMatchLocation = maxLoc
                bestWidth = scaledWidth
                bestHeight = scaledHeight
                bestScale = scale
                bestMatchResult = matchResult
                badCounter = 0
##                bestMethod = method
            else:
                badCounter += 1
            
        
        
        # increment our scale and adjust our width and height
        scale = scale + SCALE_INCREMENT
        scaledWidth  = int(templateImageWidth * scale)
        scaledHeight = int(templateImageHeight * scale)
        


    # if bestMatch == 0, that means our template was bigger than our original, ignoring
    # through the first loop entirely without doing anything. so now we make our template
    # image small enough for the original to be searched through.
    # the reason we do this after our first loop is because the first loop scales up the 
    # template image. and if our template is already too big, we can just skip that part 
    # entirely. but in all scenarios, we still wanna scale down in case we can find a better match.

    # if we skipped the first loop entirely, that means our template is bigger than our source
    # so we scale the template down to match the source"s
    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        if bestMatch == float("inf"):
            # find which side to scale off of. will be based on whichever difference is bigger
            widthDifference = templateImageWidth - sourceImageWidth
            heightDifference = templateImageHeight - sourceImageHeight

            # we scale our images, and start our scale here
            if widthDifference >= heightDifference:
                scale = sourceImageWidth / templateImageWidth
            else:
                scale = sourceImageHeight / templateImageHeight
        # just reset our variables if we didn"t need to scale up our original image
        else:
            scale = 1.0

    else:
        if bestMatch == 0:
            # find which side to scale off of. will be based on whichever difference is bigger
            widthDifference = templateImageWidth - sourceImageWidth
            heightDifference = templateImageHeight - sourceImageHeight

            # we scale our images, and start our scale here
            if widthDifference >= heightDifference:
                scale = sourceImageWidth / templateImageWidth
            else:
                scale = sourceImageHeight / templateImageHeight
        # just reset our variables if we didn"t need to scale up our original image
        else:
            scale = 1.0

    scaledWidth  = int(templateImageWidth * scale)
    scaledHeight = int(templateImageHeight * scale)

    # now we look through the image by scaling down the template
    # since we"re scaling down, we don"t have to worry about the og"s dimension limits.
    # so instead we set the limit of how many bad matches we get in succession and
    # make sure our scale doesnt go below 0
    # same code as our first loop except we increment our scale instead of decrementing

    xBoundary = sourceImageWidth * 0.05
    yBoundary = sourceImageHeight * 0.05
    badCounter = 0
    
    
    while scaledWidth > xBoundary and scaledHeight > yBoundary and badCounter < BAD_COUNTER_MAX:
        scaledGrayTemplateImage = cv2.resize(grayTemplateImage, (scaledWidth, scaledHeight))
        
        if maskImage is None:
            matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
        else:
            scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
            matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
##            matchResult, method = runAllTemplateMatchingMethods(graySourceImage, scaledGrayTemplateImage, scaledMaskImage)
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResult)

        # check to see if our current match is better than our best match
        if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
            if minVal < bestMatch:
                # if so, update our tracking variables and reset our bad counter
                bestMatch = minVal
                bestMatchLocation = minLoc
                bestWidth = scaledWidth
                bestHeight = scaledHeight
                bestScale = scale
                bestMatchResult = matchResult
                badCounter = 0
##                    bestMethod = method
            else:
                badCounter += 1
        else:
            if maxVal > bestMatch:
                # if so, update our tracking variables and reset our bad counter
                bestMatch = maxVal
                bestMatchLocation = maxLoc
                bestWidth = scaledWidth
                bestHeight = scaledHeight
                bestScale = scale
                bestMatchResult = matchResult
                badCounter = 0
##                    bestMethod = method
            else:
                badCounter += 1
            
        
        scale = scale - SCALE_INCREMENT
        scaledWidth  = int(templateImageWidth * scale)
        scaledHeight = int(templateImageHeight * scale)
    

    # get a cropped image of our matched area from the original image
    x1 = bestMatchLocation[0]
    x2 = x1 + bestWidth
    y1 = bestMatchLocation[1]
    y2 = y1 + bestHeight
    subimage = colorSourceImage[y1:y2, x1:x2]
    
    print("best match:", bestMatch)
    #print("best method:", bestMethod)
    
    return subimage, bestWidth, bestHeight, bestScale, bestMatchResult


SOURCE_IMG_NAME = "hibiki ipad"
SOURCE_IMG_PATH = SOURCE_IMG_NAME + ".png"
EQUIPMENT_SUBIMG_PATH = SOURCE_IMG_NAME + " skill.png"
STATS_SUBIMG_PATH = SOURCE_IMG_NAME + " stat.png"


def main(equipmentSubimage, scale, f):
##    student = Student()
    # load our original image
##    sourceImage = cv2.imread(sourceImagePath, cv2.IMREAD_COLOR)

##    cv2.imshow("source image", sourceImage)

    
    # multi-scale template match for our skill template
##    equipmentSubimage, equipmentSubimageWidth, equipmentSubimageHeight, equipmentSubimageScale, equipmentSubimageMatchResult = \
##                       subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
##    cv2.imshow("equipmentSubimage", equipmentSubimage)
##    cv2.imwrite(EQUIPMENT_SUBIMG_PATH, equipmentSubimage)
    

    # scaled tempalte match for our stat window
##    statsSubimage,_,_,_ = subimageScaledSearch(sourceImage, STATS_TEMPLATE_IMAGE, STATS_MASK_IMAGE, equipmentSubimageScale)
##    cv2.imshow("statsSubimage", statsSubimage)
##    cv2.imwrite(STATS_SUBIMG_PATH, statsSubimage)
    
    
    
    ### HERE WE SHOULD EXTRACT INFO FROM THE STATS WINDOW ###
    # name, level, bond, star = extractStatsInfo(statsSubimage)


    ### NEXT WE WORK ON OUR SKILL WINDOW
    # ex, basic, passive, sub = extractSkillInfo(equipmentSubimage, star)
    # ue, ue_level = extractUEInfo(equipmentSubimage, star)
    # gear1, gear2, gear3 = extractGearInfo(equipmentSubimage, level)
    
    
##    # get the result of our template search
##    maxSkillSubimage_rgb, matchWidth, matchHeight, matchResults = subimageRGBSearch(skillWindowSubimage_rgb, maxSkillTemplate_rgb)
##    cv2.imshow("maxSkillSubimage_rgb", maxSkillSubimage_rgb)
##
##    # match_result is a 2d array where each value is the tempalte"s match percentage to
##    # the original image at that given coordinate on the image. so the result at [0][0]
##    # corresponds to the template"s match percentage at (0,0) on the image.
##    # sift through our result and get the ones that are above our threshold
##    bestMatchLocations = np.where(matchResults >= 0.80)
##    filteredMatchesResults = np.extract(matchResults >= 0.80, matchResults)
##
##    # run our coordinates and results through NMS. now we should have coordinates of the 
##    # best results with little-to-no overlapping areas
##    nmsLocations = nms(filteredMatchesResults, bestMatchLocations, matchWidth, matchHeight)


    # now with nms done, we have to determine where those max"s belong to and where levels would belong to


####    gearsTiers = getGearsTiers(equipmentSubimage, scale, studentLevel)
####    print(f + " " + str(gearsTiers))
    return 1



def getStudentStats(sourceImage, imageScale):
    # given our equipment image, TM for the stats with our scale 
    statsSubimage, _, _, statsSubimageResult, _ = subimageScaledSearch(sourceImage, STATS_TEMPLATE_IMAGE, STATS_MASK_IMAGE, imageScale)
    cv2.imshow("stats sub", statsSubimage)
    
    # process our stats subimage
    processedStatsSubimage = processBondImage(statsSubimage)
##    cv2.imshow("processedStatsSubimage", processedStatsSubimage)
    
##    studentName = getStudentName(processedStatsSubimage, imageScale)
##    studentBond = getStudentBond(processedStatsSubimage1, imageScale)
##    studentLevel = getStudentLevel(processedStatsSubimage, imageScale)
##    studentStar = getStarCount(statsSubimage, STATS_STAR_MASK_IMAGE, imageScale, STAR_TEMPLATE_IMAGE, STAR_MASK_IMAGE, STAR_MATCH_THRESHOLD)

##    print(studentName, studentBond, studentStar, studentLevel)
    
    return 1



def getStudentName(processedStatsImage, imageScale):
    processedNameSubimage = cropImageWithMask(processedStatsImage, STATS_NAME_MASK_IMAGE)
    
    studentName = convertImageToString(processedNameSubimage)
    
    return studentName


def getStudentBond(processedStatsImage, imageScale):
    processedBondSubimage = cropImageWithMask(processedStatsImage, STATS_BOND_MASK_IMAGE)
    cv2.imshow("processedBondSubimage1", processedBondSubimage)
    
    startPoint = (0, 0)         # start at the first pixel
    whiteColor = (255, 255, 255, 255)   # color to flood with
    cv2.floodFill(processedBondSubimage, None, startPoint, whiteColor)
    processedBondSubimage = np.invert(processedBondSubimage)
    
    cv2.imshow("processedBondSubimage", processedBondSubimage)
    
    studentBond = convertImageToString(processedBondSubimage)

    print(studentBond)
    
    return studentBond


def getStudentLevel(processedStatsImage, imageScale):
    processedLevelSubimage = cropImageWithMask(processedStatsImage, STATS_LEVEL_MASK_IMAGE)

    studentLevel = convertImageToString(processedLevelSubimage)
    
    return studentLevel


##def getStudentStats(statsImage, imageScale):
##    processedStatsSubimage = processImage(statsImage)
##    cv2.imshow("processedStatsSubimage", processedStatsSubimage)
##    
##    studentStats = []
##    
##    for statsInfoMaskImage in STATS_INFO_MASK_IMAGES:
##        processedStatsInfoSubimage = cropImageWithMask(processedStatsSubimage, statsInfoMaskImage)
##        
##        studentStat = convertImageToString(processedStatsInfoSubimage)
##        
##        studentStats.append(studentStat)
##
##    
##
##    return studentStats



def getStarCount(sourceImage, sourceStarMaskImage, imageScale, starTemplateImage, starMaskImage, starMatchThreshold):
    # given the stats subimage, crop out the area where the stars should be using our mask
    sourceStarSubimage = cropImageWithMask(sourceImage, sourceStarMaskImage)
    
    # TM for star template in our starsSubimage
    _, starSubimageWidth, starSubimageHeight, _, starSubimageResults = subimageScaledSearch(sourceStarSubimage, starTemplateImage, starMaskImage, imageScale)
    
    # filter our results to be below the threshold and grab the coordinates of those matches
    filteredStarResults = np.extract(starSubimageResults <= starMatchThreshold, starSubimageResults)
    filteredStarLocations = np.where(starSubimageResults <= starMatchThreshold)
    
    # filter out the results even more to remove matches that overlap with our best matches
    nmsStarLocations, nmsStarResults = nms(filteredStarResults, filteredStarLocations, starSubimageWidth, starSubimageHeight, STAR_OVERLAP_THRESHOLD)
    
##    for index in range(len(nmsStarLocations)):
##        print(nmsStarLocations[index])
##        print(nmsStarResults[index])
##        color = (0, index/len(nmsStarLocations) * 255, 0)
##        cv2.rectangle(sourceStarSubimage, nmsStarLocations[index], (nmsStarLocations[index][0] + starSubimageWidth, nmsStarLocations[index][1] + starSubimageHeight), color, 1)
##
##    cv2.imshow(str(time.time()), sourceStarSubimage)
##    sourceStarSubimage = scaleImage(sourceStarSubimage, 5)
##    cv2.imshow(str(time.time()), sourceStarSubimage)
    
    # the amount of nms-filtered results is the amount of stars we have
    studentStar = len(nmsStarResults)
    
    return studentStar


directory = "student example ipad"
scale = 1.0
##for fileName in os.listdir(directory):
##    f = os.path.join(directory, fileName)
##
##    
##    sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)
####    _, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
##
##    
##    print(f)
##    getStudentStats(sourceImage, scale)

sourceImage = cv2.imread(r"student example ipad\nodoka ipad.png", cv2.IMREAD_COLOR)
getStudentStats(sourceImage, scale)
    

######### STAT READER #########
##hibikiStats = cv2.imread("maki stats.png", cv2.IMREAD_COLOR)
##processedHibiki = processImage(hibikiStats)
##
##hibikiStars = processImage(hibikiStars)
##hibikiHeart = processImage(hibikiHeart)
##hibikiLevel = processImage(hibikiLevel)
##hibikiName  = processImage(hibikiName)
##
##startPoint = (0, 0)         # start at the first pixel
##newColor = (255, 255, 255, 255)   # color to flood with
##cv2.floodFill(hibikiHeart, None, startPoint, newColor)
##hibikiHeart = np.invert(hibikiHeart)
##
##
##cv2.imshow("processedHibiki", processedHibiki)
##print(convertImageToString(processedHibiki))
##
##cv2.imshow("hibikiLevel", hibikiLevel)
##print("hibikiLevel: ", convertImageToString(hibikiLevel))
##
##cv2.imshow("hibikiHeart", hibikiHeart)
##print("hibikiHeart: ", convertImageToString(hibikiHeart))
##
##cv2.imshow("hibikiStars", hibikiStars)
##print("hibikiStars: ", convertImageToString(hibikiStars))
##
##cv2.imshow("hibikiName", hibikiName)
##print("hibikiName: ", convertImageToString(hibikiName))
##
##hibikiHeartName = cv2.bitwise_or(hibikiName, hibikiHeart)
##cv2.imshow("hibikiHeartName", hibikiHeartName)
##print("hibikiHeartName: ", convertImageToString(hibikiHeartName))



######### getting UE stars #############
##sourceImage = cv2.imread("akane ipad.png", cv2.IMREAD_COLOR)
##equipmentTemplateImage = cv2.imread("skill template.png", cv2.IMREAD_COLOR)
##equipmentMaskImage = cv2.imread("skill mask.png", cv2.IMREAD_GRAYSCALE)
##
##startTime = time.time()
##equipmentSubimage, _, _, equipmentSubimageScale, _ = subimageMultiScaleSearch(sourceImage, equipmentTemplateImage, equipmentMaskImage)
##cv2.imshow("equipmentSubimage", equipmentSubimage)
##endTime = time.time()
##print("Time: ", (endTime-startTime))
##
##ueTemplateImage = cv2.imread("ue template.png", cv2.IMREAD_COLOR)
##ueMaskImage = cv2.imread("ue mask.png", cv2.IMREAD_GRAYSCALE)
##ueSubimage, _, _, _ = subimageScaledSearch(equipmentSubimage, ueTemplateImage, ueMaskImage, equipmentSubimageScale)
##cv2.imshow("ueSubimage", ueSubimage)
##
##ueStarsMaskImage = cv2.imread("ue stars mask.png", cv2.IMREAD_GRAYSCALE)
##ueStarsSubimage = cropImageWithMask(ueSubimage, ueStarsMaskImage)
##cv2.imshow("ueStarsSubimage", ueStarsSubimage)
##
##
##
##cv2.imwrite("akane ipad ue stars subimage.png", ueStarsSubimage)
##ueStarsSubimage = cv2.imread("akane ipad ue stars subimage.png", cv2.IMREAD_COLOR)
##
##
##
##ueStarTemplate = cv2.imread("ue star template.png", cv2.IMREAD_COLOR)
##ueStarMask = cv2.imread("ue star mask.png", cv2.IMREAD_GRAYSCALE)
####ueStarMask = createMaskFromTransparency(ueStarTemplate)
##ueStarSubimage, matchWidth, matchHeight, ueStarScale, ueStarResult = subimageMultiScaleSearch(ueStarsSubimage, ueStarTemplate, ueStarMask)#, equipmentSubimageScale)
##
##cv2.imshow("ueStarSubimage", ueStarSubimage)
##ueStarSubimage = cv2.resize(ueStarSubimage, (ueStarSubimage.shape[1] *5,ueStarSubimage.shape[0] *5))
##cv2.imshow("resueStarSubimage", ueStarSubimage)
##
##bestMatchLocations = np.where(ueStarResult <= UE_STAR_THRESHOLD)
##filteredMatchesResults = np.extract(ueStarResult <= UE_STAR_THRESHOLD, ueStarResult)
##
### run our coordinates and results through NMS. now we should have coordinates of the 
### best results with little-to-no overlapping areas
##nmsLocations, nmsResults = nms(TEMPLATE_MATCH_METHOD, filteredMatchesResults, bestMatchLocations, matchWidth, matchHeight)
##
##
##for index in range(len(nmsLocations)):
####    print(nmsLocations[index])
####    print(nmsResults[index])
##    color = (0, index/len(nmsLocations) * 255, 0)
##    cv2.rectangle(ueStarsSubimage, nmsLocations[index], (nmsLocations[index][0] + matchWidth, nmsLocations[index][1] + matchHeight), color, 1)
##
##cv2.imshow("ueStarsSubimage", ueStarsSubimage)
##ueStarsSubimage = cv2.resize(ueStarsSubimage, (ueStarsSubimage.shape[1] *5,ueStarsSubimage.shape[0] *5))
##cv2.imshow("resueStarsSubimage", ueStarsSubimage)
##cv2.imwrite("akane ipad ue stars boxed subimage.png", ueStarsSubimage)
##ueStarCount = len(nmsLocations)
###print(ueStarCount)



def getGearsTiers(equipmentImage, imageScale, studentLevel):
    # given our equipment image, TM for the gears 
    gearsSubimage, _, _, gearsSubimageResult, _ = subimageScaledSearch(equipmentImage, GEARS_TEMPLATE_IMAGE, GEARS_MASK_IMAGE, imageScale)
##    cv2.imshow("gearsSubimage",gearsSubimage)
    
    
    # get the gear slot count
    gearCount = len(GEAR_SLOT_LEVEL_REQUIREMENTS)

    # array to keep track of our student's gear's tiers
    gearsTiers = np.zeros(gearCount, np.int)
    
    # go through all of our individual gears
    for index in range(gearCount):
        # get the current gear's level req
        gearSlotLevelRequirement = GEAR_SLOT_LEVEL_REQUIREMENTS[index]
        
        # if student level isn't high enough, we break
        if studentLevel < gearSlotLevelRequirement:
            break
        
        # get the respective gear mask
        gearMaskImage = GEAR_MASK_IMAGES[index]
        
        # get the gear subimage
        gearSubimage = cropImageWithMask(gearsSubimage, gearMaskImage)
##        cv2.imshow("gearSubimage",gearSubimage)
        
        # check if the current gear slot has anything equipped
        isGearEquipped = checkGearEquipped(gearSubimage, imageScale)
        
        # if a gear is equipped, we get the tier
        if isGearEquipped:
            # get the tier of the image and add it to our list
            gearTier = getGearTier(gearSubimage, imageScale)
            gearsTiers[index] = gearTier
    
    
    # return our tiers
    return gearsTiers


def checkGearEquipped(gearImage, imageScale):
    # given the specific gear image (gear1, gear2, or gear3), we get the subimage of where the "E" should be
    # the "E" represents whether there is a gear equipped in the slot
    gearESubimage = cropImageWithMask(gearImage, GEAR_E_MASK_IMAGE)
    
    # template match for the "E"
    eSubimage, _, _, eSubimageResult, _ = subimageScaledSearch(gearESubimage, E_TEMPLATE_IMAGE, E_MASK_IMAGE, imageScale)
##    cv2.imshow(str(time.time()), eSubimage)
##    print(eSubimageResult)
##    return True
    
    # if the "E" match result is lower than our threshold, that means the "E" was found and there is a gear equipped
    if eSubimageResult < E_MATCH_THRESHOLD:
        return True
    else:
        return False


def getGearTier(gearImage, imageScale):
    # given the specific gear image (gear1, gear2, or gear3), we get the subimage of where the tier should be
    gearTierSubimage = cropImageWithMask(gearImage, GEAR_TIER_MASK_IMAGE)
    
    # variables to keep track of which tier template matches best with our image
    bestTierResult = float("inf")
    bestTierSubimage = gearImage
    bestTier = 0
    
    # counter/tracker for which tier we're currently using 
    currentTier = 1
    
    # go through all the tier templates and see which one matches best 
    for tierTemplateImage in TIER_TEMPLATE_IMAGES:
        # template match the current tier template in our subimage
        tierSubimage, _, _, tierSubimageResult, _ = subimageScaledSearch(gearTierSubimage, tierTemplateImage, TIER_MASK_IMAGE, imageScale)
        
        # check to see if our current match is better than our best match
        # update our variables if so
        if tierSubimageResult < bestTierResult:
            bestTierResult = tierSubimageResult
            bestTierSubimage = tierSubimage
            bestTier = currentTier
        
        # increment our tier tracker
        currentTier += 1
    
##    cv2.imshow(str(time.time()), bestTierSubimage)
##    print(bestTierResult)
    # return our best tier
    return bestTier



##directory = "student example ipad"
##sourceImage = cv2.imread(r"student example ipad\airi ipad.png", cv2.IMREAD_COLOR)
#_, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
##scale = 1
##
##for fileName in os.listdir(directory):
##    f = os.path.join(directory, fileName)
##    sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)
##    subimage, _, _, _, _ = subimageScaledSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE, scale)
##
##    main(subimage, scale, f)
