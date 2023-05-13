import os
import cv2
import time
import random
import numpy as np
from pytesseract import pytesseract


MAX_HIT = 0
MIN_MISS = float("inf")

NAME_ADDITION_IMAGE = cv2.imread("name addition.png", cv2.IMREAD_GRAYSCALE)



BOND_CHEAT_SHEET_IPAD = [[2,0],[2,0],[1,4],[2,2],[1,6],[2,2],[1,9],[1,9],[1],[1,3],[2,2],[2,0],[2,0],[2,0],[1,8],[1,7],[1,4],[2,0],[2,0],[1,2],[2,0],[2,0],[2,0],[2,0],[2,0],[1,8],[1,5],[1,7],[4]]
BOND_CHEAT_SHEET_ARRAY = [[2,0],[2,0],[9],[2,0],[2,0],[1,4],[2,2],[1,6],[2,2],[1,9],[1,9],[6],[6],[2],[2],[1],[1,3],[2,2],[2,0],[2,0],[2,0],[1,8],[2,4],[2,4],[8],[1,7],[1,4],[2,0],[2,3],[8],\
                          [2,0],[1,2],[2,0],[2,0],[1,7],[2,0],[2,0],[2,0],[1,8],[1,5],[1,7],[4]]
BOND_CHEAT_SHEET = [20,20,9,20,20,14,22,16,22,19,19,6,6,2,2,1,13,22,20,20,20,18,24,24,8,17,14,20,23,8,20,12,20,20,17,20,20,20,18,15,17,4]

SCALE_CHEAT_SHEET = [ 1.0, 0.8999999999999999, 1.0, 0.8999999999999999, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0, 1.0, 1.0, \
                      0.8899999999999999, 0.8899999999999999, 0.8899999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, \
                      1.0, 1.0, 0.8899999999999999, 0.5086363636363634, 0.8899999999999999, 1.0, 1.0, 1.0, \
                      0.8899999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
LEVEL_CHEAT_SHEET = [1,1,1,80,9,3,80,35,75,1,5,80,1,1,1,1,75,80,10,71,7,38,80,80,1,6,15,1,80,76,45,70,\
                     35,8,75,2,35,12,4,1,75,75]
LEVEL_CHEAT_SHEET_ARRAY = [[1],[1],[1],[8,0],[9],[3],[8,0],[3,5],[7,5],[1],[5],[8,0],[1],[1],[1],[1],[7,5],[8,0],[1,0],[7,1],[7],[3,8],[8,0],[8,0],[1],\
                           [6],[1,5],[1],[8,0],[7,6],[4,5],[7,0],[3,5],[8],[7,5],[2],[3,5],[1,2],[4],[1],[7,5],[7,5]]
STAR_CHEAT_SHEET = [5,4,3,5,3,3,5,3,5,4,4,5,2,1,1,1,3,5,3,4,4,3,5,5,3,5,4,3,5,3,4,3,4,5,5,3,4,5,4,4,3,3]

STUDENT_CHEAT_SHEET = [[1.0, "Airi", 20, 5, 1], [0.8999999999999999, "Airi", 20, 4, 1], [1.0, "Akane (Bunny)", 9, 3, 1], [0.8999999999999999, "Akane", 20, 5, 80], \
                       [1.0, "Ayane", 20, 3, 9], [1.0, "Ayane (Swimsuit)", 14, 3, 3], [1.0, "Cherino", 22, 5, 80], [1.0, "Chihiro", 16, 3, 35],\
                       [1.0, "Hanae", 22, 5, 75], [1.0, "Hanako", 19, 4, 1], [1.0, "Hanako", 19, 4, 1]]

COUNTER = 0



PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEAR_X_OFFSET = 4
GEAR_Y_OFFSET = 6

BAD_COUNTER_MAX = 150000
SCALE_INCREMENT = 0.01



## recorded the highest correct guess and lowest wrong guess
# min miss: 0.04736049473285675
# max hit: 0.038083869963884354
BOND_MATCH_THRESHOLD = 0.0427

## recorded the highest value in overlapPercentages of values that were below 0.50 and the lowest value of those that were above 0.50
## should remember that overlap is determined by the match threshold because match threshold is the initial filter
# min miss: 0.9212121212121213
# max hit: 0.4090909090909091
BOND_OVERLAP_THRESHOLD = 0.665


## same process as bond_match.
## even with the match threshold being high, the nms should filter out the bad eggs... hopefully...
# min miss: 0.060845356434583664
# max hit: 0.061924394220113754
LEVEL_MATCH_THRESHOLD = 0.07

# same process and bond_overlap
# min miss: 0.8625
# max hit: 0.46153846153846156
LEVEL_OVERLAP_THRESHOLD = 0.662


## find the overlap threshold first then the match threshold.
## i found the max-correct overlap and set my threshold somewhere above that.
## then filtering with my new overlap threshold, i got the max-correct match and the min-incorrect match and made the match thresh.
# min_miss: 0.7591266
# max hit:  0.019280406
STAR_MATCH_THRESHOLD = 0.1

# max hit: 0.3488372093023256
STAR_OVERLAP_THRESHOLD = 0.4


## same process as star_match
# min miss: 0.044682324
# max hit: 0.010489287
STAR_UE_MATCH_THRESHOLD = 0.0276

# max hit: 0.22727273
STAR_UE_OVERLAP_THRESHOLD = 0.3

# my one wrong match: 0.04139775037765503
# max_hit: 0.005240572150796652
# average: 0.0013653720586955
E_UE_MATCH_THRESHOLD = 0.01


# min miss:     0.612736701965332
# average miss: 0.72706709057093
# max hit:      0.1853787750005722
# average hit:  0.036196497934567
E_MATCH_THRESHOLD = 0.3


# min miss:     0.0605535731
# average miss: 0.069173962343484
# max hit:      0.0261527728
# average hit:  0.0080085342507
TIER_MATCH_THRESHOLD = 0.04






# ccorr is bad because "we're dealing with discrete digital signals that have a
# defined maximum value (images), that means that a bright white patch of the image will basically
# always have the maximum correlation" and ccoeff pretty much fixes that problem.
# but for some reason ccoeff has been returning infinity in some cases... so yeah guess not that.
# sqdiff calculates the intensity in DIFFERENCE of the images. so you wanna look for the min if using sqdiff.
# also you want to use NORMED functions when template matching with templates of different sizes (ew're doing multiscale)
TEMPLATE_MATCH_METHOD = cv2.TM_SQDIFF_NORMED

# array of what min level you need to be to unlock the respective gear slot. slot1=lvl0, slot2=lvl15, slot3=lvl35
GEAR_SLOT_LEVEL_REQUIREMENTS = [0, 15, 35]
UE_SLOT_STAR_REQUIREMENT = 5


# STATS templates and masks
STATS_TEMPLATE_IMAGE = cv2.imread("stats template.png", cv2.IMREAD_COLOR)
STATS_MASK_IMAGE = cv2.imread("stats mask.png", cv2.IMREAD_GRAYSCALE)

STATS_NAME_MASK_IMAGE = cv2.imread("stats name mask.png", cv2.IMREAD_GRAYSCALE)
STATS_BOND_MASK_IMAGE = cv2.imread("stats bond mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_MASK_IMAGE = cv2.imread("stats level mask.png", cv2.IMREAD_GRAYSCALE)


BOND_TEMPLATE_IMAGES = []
BOND_0_TEMPLATE_IMAGE = cv2.imread("bond 0 template.png", cv2.IMREAD_COLOR)
BOND_1_TEMPLATE_IMAGE = cv2.imread("bond 1 template.png", cv2.IMREAD_COLOR)
BOND_2_TEMPLATE_IMAGE = cv2.imread("bond 2 template.png", cv2.IMREAD_COLOR)
BOND_3_TEMPLATE_IMAGE = cv2.imread("bond 3 template.png", cv2.IMREAD_COLOR)
BOND_4_TEMPLATE_IMAGE = cv2.imread("bond 4 template.png", cv2.IMREAD_COLOR)
BOND_5_TEMPLATE_IMAGE = cv2.imread("bond 5 template.png", cv2.IMREAD_COLOR)
BOND_6_TEMPLATE_IMAGE = cv2.imread("bond 6 template.png", cv2.IMREAD_COLOR)
BOND_7_TEMPLATE_IMAGE = cv2.imread("bond 7 template.png", cv2.IMREAD_COLOR)
BOND_8_TEMPLATE_IMAGE = cv2.imread("bond 8 template.png", cv2.IMREAD_COLOR)
BOND_9_TEMPLATE_IMAGE = cv2.imread("bond 9 template.png", cv2.IMREAD_COLOR)
BOND_TEMPLATE_IMAGES.append(BOND_0_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_1_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_2_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_3_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_4_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_5_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_6_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_7_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_8_TEMPLATE_IMAGE)
BOND_TEMPLATE_IMAGES.append(BOND_9_TEMPLATE_IMAGE)

BOND_MASK_IMAGES = []
BOND_0_MASK_IMAGE = cv2.imread("bond 0 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_1_MASK_IMAGE = cv2.imread("bond 1 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_2_MASK_IMAGE = cv2.imread("bond 2 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_3_MASK_IMAGE = cv2.imread("bond 3 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_4_MASK_IMAGE = cv2.imread("bond 4 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_5_MASK_IMAGE = cv2.imread("bond 5 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_6_MASK_IMAGE = cv2.imread("bond 6 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_7_MASK_IMAGE = cv2.imread("bond 7 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_8_MASK_IMAGE = cv2.imread("bond 8 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_9_MASK_IMAGE = cv2.imread("bond 9 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_MASK_IMAGES.append(BOND_0_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_1_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_2_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_3_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_4_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_5_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_6_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_7_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_8_MASK_IMAGE)
BOND_MASK_IMAGES.append(BOND_9_MASK_IMAGE)


LEVEL_TEMPLATE_IMAGES = []
LEVEL_0_TEMPLATE_IMAGE = cv2.imread("level 0 template.png", cv2.IMREAD_COLOR)
LEVEL_1_TEMPLATE_IMAGE = cv2.imread("level 1 template.png", cv2.IMREAD_COLOR)
LEVEL_2_TEMPLATE_IMAGE = cv2.imread("level 2 template.png", cv2.IMREAD_COLOR)
LEVEL_3_TEMPLATE_IMAGE = cv2.imread("level 3 template.png", cv2.IMREAD_COLOR)
LEVEL_4_TEMPLATE_IMAGE = cv2.imread("level 4 template.png", cv2.IMREAD_COLOR)
LEVEL_5_TEMPLATE_IMAGE = cv2.imread("level 5 template.png", cv2.IMREAD_COLOR)
LEVEL_6_TEMPLATE_IMAGE = cv2.imread("level 6 template.png", cv2.IMREAD_COLOR)
LEVEL_7_TEMPLATE_IMAGE = cv2.imread("level 7 template.png", cv2.IMREAD_COLOR)
LEVEL_8_TEMPLATE_IMAGE = cv2.imread("level 8 template.png", cv2.IMREAD_COLOR)
LEVEL_9_TEMPLATE_IMAGE = cv2.imread("level 9 template.png", cv2.IMREAD_COLOR)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_0_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_1_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_2_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_3_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_4_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_5_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_6_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_7_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_8_TEMPLATE_IMAGE)
LEVEL_TEMPLATE_IMAGES.append(LEVEL_9_TEMPLATE_IMAGE)

LEVEL_MASK_IMAGES = []
LEVEL_0_MASK_IMAGE = cv2.imread("level 0 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_1_MASK_IMAGE = cv2.imread("level 1 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_2_MASK_IMAGE = cv2.imread("level 2 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_3_MASK_IMAGE = cv2.imread("level 3 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_4_MASK_IMAGE = cv2.imread("level 4 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_5_MASK_IMAGE = cv2.imread("level 5 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_6_MASK_IMAGE = cv2.imread("level 6 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_7_MASK_IMAGE = cv2.imread("level 7 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_8_MASK_IMAGE = cv2.imread("level 8 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_9_MASK_IMAGE = cv2.imread("level 9 mask.png", cv2.IMREAD_GRAYSCALE)
LEVEL_MASK_IMAGES.append(LEVEL_0_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_1_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_2_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_3_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_4_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_5_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_6_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_7_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_8_MASK_IMAGE)
LEVEL_MASK_IMAGES.append(LEVEL_9_MASK_IMAGE)


STATS_STAR_MASK_IMAGE = cv2.imread("stats star mask.png", cv2.IMREAD_GRAYSCALE)
STAR_TEMPLATE_IMAGE = cv2.imread("star template.png", cv2.IMREAD_COLOR)
STAR_MASK_IMAGE = cv2.imread("star mask.png", cv2.IMREAD_GRAYSCALE)


# EQUIPMENT templates and masks
EQUIPMENT_TEMPLATE_IMAGE = cv2.imread("equipment template.png", cv2.IMREAD_COLOR)
EQUIPMENT_MASK_IMAGE = cv2.imread("equipment mask.png", cv2.IMREAD_GRAYSCALE)


# SKILLS templates and masks
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


# UE templates and masks
UE_TEMPLATE_IMAGE = cv2.imread("ue template.png", cv2.IMREAD_COLOR)
UE_MASK_IMAGE = cv2.imread("ue mask.png", cv2.IMREAD_GRAYSCALE)
UE_E_MASK_IMAGE = cv2.imread("ue E mask.png", cv2.IMREAD_GRAYSCALE)
UE_STAR_MASK_IMAGE = cv2.imread("ue star mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_MASK_IMAGE = cv2.imread("ue level mask.png", cv2.IMREAD_GRAYSCALE)

STAR_UE_TEMPLATE_IMAGE = cv2.imread("star ue template.png", cv2.IMREAD_COLOR)
STAR_UE_MASK_IMAGE = cv2.imread("star ue mask.png", cv2.IMREAD_GRAYSCALE)

E_UE_TEMPLATE_IMAGE = cv2.imread("E ue template.png", cv2.IMREAD_COLOR)
E_UE_MASK_IMAGE = cv2.imread("E ue mask.png", cv2.IMREAD_GRAYSCALE)



# GEARS tempaltes and masks
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


# given an image, convert it to a string through pytesseract
def convertImageToString(image):
    # load in the tesseract
    pytesseract.tesseract_cmd = PATH_TO_TESSERACT
    
    # convert the iamge to a stirng
    imageToString = pytesseract.image_to_string(image)
    
    # return the string
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
    
    # scale our dimensions according to the given scale
    scaledWidth = int(imageWidth * scale)
    scaledHeight = int(imageHeight * scale)
    
    # resize our target image to the new dimensions
    scaledImage = cv2.resize(imageToResize, (scaledWidth, scaledHeight))
    
    # return the scaled image
    return scaledImage


# given an image and a mask, crop the source within the area of the mask
# intended to be used with masks that only have 1 rectangular area 
def cropImageWithMask(sourceImage, maskImage):
    # resize our mask to our source
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
    # get shape of our image
    imageChannelCount = image.shape[2]
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]

    # if our image is only bgr (3 channels with no alpha), then we don't need to make a mask.
    # because even using an all-white mask (essentailly same as no mask), causes the runtime to
    # be about twice as long. so better to rock none
    if imageChannelCount == 3:
        return None
    
    ## create an image with its values being the max. then we adjust the pixels according to the alpha channel
    # start out with an all-white image
    maskImage = np.ones((imageHeight, imageWidth), np.uint8) * 255
    
    # turn the img's transparency array into a scalar array.
    # an image's transparency is on index 3, and the max value is 255 (for completely visible)
    imageAlphaChannel = image[:,:,3] 
    imageAlphaScalar = imageAlphaChannel / 255.0
    
    # apply the scalar to our white image
    maskImage[:,:] = imageAlphaScalar[:,:] * maskImage[:,:]
    
    # return the created mask image
    return maskImage


# given two images, and optional offsets, combine two images together,
# blending in their transparencies and return it
def overlapTransparentImages(bgImage, fgImage, xOffset = 0, yOffset = 0):
    # get dimensions of our subarea, which is the foreground image (fgImage)
    fgImageWidth = fgImage.shape[1]
    fgImageHeight = fgImage.shape[0]
    
    # determine the area we're working with. offset + fg dimensions
    x1 = xOffset
    x2 = x1 + fgImageWidth
    y1 = yOffset 
    y2 = y1 + fgImageHeight
    
    # get the alpha channel of our fgimage and turn it into a scalar. turns 0-255 into 0-1
    fgImageAlphaChannel = fgImage[:,:,3]
    fgImageAlphaScalar = fgImageAlphaChannel / 255.0
    
    # calculate what the background image's (bgimage) alpha should be. it is fgimage's complement 
    bgImageAlphaScalar = 1.0 - fgImageAlphaScalar
    
    # initialize our overlapped image. the base will be the bg image
    overlappedImage = bgImage.copy()
    
    # apply the scalar to each color channel, bgr
    for colorChannel in range (0, 3):
        # grab the color channel and apply the scalar to said channel on the subarea in the bg image
        bgImageColorChannel = bgImage[y1:y2, x1:x2, colorChannel]  # color channel
        bgImageColor = bgImageAlphaScalar * bgImageColorChannel    # alpha applied to color channel
        
        # then apply the scalar to all of the fg image
        fgImageColorChannel = fgImage[:, :, colorChannel]
        fgImageColor = fgImageAlphaScalar * fgImageColorChannel
        
        # combine the colors from both images together in the subarea
        overlappedImage[y1:y2, x1:x2, colorChannel] = bgImageColor + fgImageColor
    
    # return the overlapped image
    return overlappedImage


# crop out the name subarea from the statsImage. then we process it by thresholding. then we concatenate
# an image with "Name: " to the left of it to help pytesseract read better. then return the concatenated image
def processNameImage(statsImage, imageScale):
    # crop the source according to our mask
    nameImage = cropImageWithMask(statsImage, STATS_NAME_MASK_IMAGE)
    
    # process our nameImage
    grayNameImage = cv2.cvtColor(nameImage, cv2.COLOR_BGR2GRAY)
    processedNameImage = cv2.threshold(grayNameImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # scale our addition image accordingly and then resize our its height to match nameImage's
    resizedNameAdditionImage = scaleImage(NAME_ADDITION_IMAGE, imageScale)
    resizedNameAdditionImage = cv2.resize(resizedNameAdditionImage, (resizedNameAdditionImage.shape[1], nameImage.shape[0]))
    
    # then we concatenate them, putting the two images side by side
    concatenatedImage = cv2.hconcat([resizedNameAdditionImage, processedNameImage])
    
    # and return the concatenated image
    return concatenatedImage


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
##    blankImage1 = cropImageWithMask(blankImage1, STATS_BOND_NAME_MASK_IMAGE)
##    blankImage2 = cropImageWithMask(blankImage2, STATS_BOND_NAME_MASK_IMAGE)
##    blankImage3 = cropImageWithMask(blankImage3, STATS_BOND_NAME_MASK_IMAGE)
##    blankImage4 = cropImageWithMask(blankImage4, STATS_BOND_NAME_MASK_IMAGE)
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
    
    # return the error and the difference of the subimage
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


# from an array of results, locations, and sizes, we filter the results that
# overlap with our best matches, guaranteeing unique results
def nms(matchResults, matchLocations, matchWidth, matchHeight, overlapThreshold):
    # unpack our arrays
    x1MatchCoordinates = matchLocations[:,0]
    x2MatchCoordinates = x1MatchCoordinates + matchWidth
    y1MatchCoordinates = matchLocations[:,1]
    y2MatchCoordinates = y1MatchCoordinates + matchHeight
    
    # determine which order to run through the results/coordinates.
    # we do it based on the results, starting with the best match.
    # we create a dedicated "order" array instead of just sorting the results array itself.
    # this is because we're working with other arrays (the coordinates and sizes array), so
    # it's easier than having to work with all arrays simultaneously by sorting, removing elements, etc.
    indexOrder = np.argsort(matchResults)[::1]
    
    # create an array to store our overlap percentages
    overlap = []
    
    # array to store the coordinates of our best matches, filtering out overlappers
    nmsResults = []
    nmsLocations = []
    
    # array to keep track of the nms-filtered index order we traversed
    nmsOrder = []
    
    
    # go through all our boxes starting with best match.
    # grab our best match and store it into out filtered list.
    # then compare it to the rest of our boxes and record their overlap iwth our best match.
    # if any have overlap over the threshhold, we delete those values from our index list, ignoring them as unique matches
    while len(indexOrder) > 0:
        # get the index to work with. should be our current highest match result
        # which should be the first item in the list.
        bestMatchIndex = indexOrder[0]
        
        # get the location of the current best match and add it to our bestMatchLocations list
        x1BestMatchCoordinate = x1MatchCoordinates[bestMatchIndex]
        x2BestMatchCoordinate = x2MatchCoordinates[bestMatchIndex]
        y1BestMatchCoordinate = y1MatchCoordinates[bestMatchIndex]
        y2BestMatchCoordinate = y2MatchCoordinates[bestMatchIndex]
        
        nmsResults.append(matchResults[bestMatchIndex])
        nmsLocations.append((x1BestMatchCoordinate, y1BestMatchCoordinate))
        
        # get its area
        bestMatchWidth = x2BestMatchCoordinate - x1BestMatchCoordinate
        bestMatchHeight = y2BestMatchCoordinate - y1BestMatchCoordinate
        bestMatchArea = bestMatchWidth * bestMatchHeight
        
        
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
        overlapPercentages = overlapAreas / bestMatchArea
        
        # delete the indices of matches where the overlap is over a certain threshold.
        # in this case, we delete the entries of the indicies so we don't
        # iterate over them later.
        indexDelete = np.where(overlapPercentages > overlapThreshold)[0]
        indexOrder = np.setdiff1d(indexOrder, indexDelete, True)
        
        # we add our current index to our orders array
        nmsOrder.append(bestMatchIndex)
    
    
    # reshape our array to represent (x,y) coordiantes
    nmsLocations = np.reshape(nmsLocations, (-1, 2))
    
    # the amount of results is the amount of unique items
    nmsCount = len(nmsResults)
    
##    print("nms: ", nmsResults)
    

    
##    global MAX_HIT
##    global MIN_MISS

####    #### GET MAX AND MIN OVERLAP ####
####    x1MatchCoordinates = nmsLocations[:,0]
####    x2MatchCoordinates = x1MatchCoordinates + matchWidth
####    y1MatchCoordinates = nmsLocations[:,1]
####    y2MatchCoordinates = y1MatchCoordinates + matchHeight
####    allOverlapPercentages = []
####    
####    for index in range(nmsCount):
####        index = nmsOrder[index]
####
####        x1BestMatchCoordinate = x1MatchCoordinates[index]
####        x2BestMatchCoordinate = x2MatchCoordinates[index]
####        y1BestMatchCoordinate = y1MatchCoordinates[index]
####        y2BestMatchCoordinate = y2MatchCoordinates[index]
####        
####        # get its area
####        bestMatchWidth = x2BestMatchCoordinate - x1BestMatchCoordinate
####        bestMatchHeight = y2BestMatchCoordinate - y1BestMatchCoordinate
####        bestMatchArea = bestMatchWidth * bestMatchHeight
####        
####        x1OverlapCoordinates = np.maximum(x1BestMatchCoordinate, x1MatchCoordinates)
####        x2OverlapCoordinates = np.minimum(x2BestMatchCoordinate, x2MatchCoordinates)
####        y1OverlapCoordinates = np.maximum(y1BestMatchCoordinate, y1MatchCoordinates)
####        y2OverlapCoordinates = np.minimum(y2BestMatchCoordinate, y2MatchCoordinates)
####        
####        ## calculate the area of overlap
####        # we do a max of 0, because if you have a negative edges, that means there isn't 
####        # any overlap and actually represents the area of empty space between the two matches.
####        # and if you have 2 negative edges, the area will
####        # still be positive, resulting in a false positive.
####        overlapWidths  = np.maximum(0, x2OverlapCoordinates - x1OverlapCoordinates)
####        overlapHeights = np.maximum(0, y2OverlapCoordinates - y1OverlapCoordinates)
####        overlapAreas = overlapWidths * overlapHeights
####
####        overlapPercentages = overlapAreas / bestMatchArea
####
####        allOverlapPercentages.append(overlapPercentages)
####        
####        # here i wanna go through the matrix. change max_hit from the area within row AND col < star_cheat[counter]
####        print(overlapPercentages)
####    
####    allOverlapPercentages = np.array(allOverlapPercentages)
####    allOverlapPercentages = np.reshape(allOverlapPercentages, (nmsCount, nmsCount))
####    
####    starCount = STAR_CHEAT_SHEET[COUNTER]
####    print("star:", starCount)
####
####    overlapHits = allOverlapPercentages[:starCount, :starCount]
####    overlapHits = overlapHits.flatten()
####
####    overlapMisses = np.setdiff1d(allOverlapPercentages, overlapHits)
####    print("hits:", overlapHits)
####    print("miss:", overlapMisses)
####    overlapHits = np.setdiff1d(overlapHits, [1.0])
####    overlapMisses = np.setdiff1d(overlapMisses, [0.0])
####    if len(overlapHits) > 0:
####        if max(overlapHits) > MAX_HIT:
####            MAX_HIT = max(overlapHits)
####    if len(overlapMisses) > 0:
####        if min(overlapMisses) < MIN_MISS:
####            MIN_MISS = min(overlapMisses)


##    starCount = STAR_CHEAT_SHEET[COUNTER]
##
##    if len(nmsResults) > 0:
##        if max(nmsResults[:starCount]) > MAX_HIT:
##            MAX_HIT = max(nmsResults[:starCount])
##
##        if starCount < nmsCount:
##            if min(nmsResults[starCount:]) < MIN_MISS:
##                MIN_MISS = min(nmsResults[starCount:])
    
    
##    if max(nmsResults[:starCount]) > MAX_HIT:
##        MAX_HIT = max(nmsResults[:starCount])
##
##    minValll = min(np.delete(matchResults, nmsOrder[:starCount]))
##    if minValll < MIN_MISS:
##        MIN_MISS = minValll
        
    
    # return our values
    return nmsResults, nmsLocations, nmsCount, nmsOrder


# filter our results and locations through a threshold, then further filter them through nms
def filterResultsAndLocations(imageResults, imageWidth, imageHeight, matchThreshold, overlapThreshold):
    # initially filter our results through our match threshold and grab the locations
    filteredResults = np.extract(imageResults <= matchThreshold, imageResults)
    (filteredYLocations, filteredXLocations) = np.where(imageResults <= matchThreshold)
    
    # np.where returns two arrays, first array of the row location and then the second array of the column locations.
    # we want to format it instead as (col, row) values, aka (x, y) values
    filteredLocations = np.column_stack((filteredXLocations, filteredYLocations))
    
    # filter out the results even more to remove overlapping matches
    nmsResults, nmsLocations, nmsCount, nmsOrder = nms(filteredResults, filteredLocations, imageWidth, imageHeight, overlapThreshold)
    
    # return our results
    return nmsResults, nmsLocations, nmsCount, nmsOrder


def drawBoxes(sourceImage, templateWidth, templateHeight, nmsLocations):
    for index in range(len(nmsLocations)):
        colorr = np.array([0, 0, 255]) * index / len(nmsLocations)
        location = nmsLocations[index]
        cv2.rectangle(sourceImage, (location), (location[0] + templateWidth, location[1] + templateHeight), colorr, 1)

    cv2.imshow(str(time.time()), scaleImage(sourceImage, 5.0))

    return 0


# template match given a scale. we scale the template and optional mask, then TM. keep the source as is.
# used for when you know what the scale of the template will be in your source image.
def subimageScaledSearch(colorSourceImage, imageScale, colorTemplateImage, maskImage = None):
    # make a grayscale copy of the main image
    graySourceImage = cv2.cvtColor(colorSourceImage, cv2.COLOR_BGR2GRAY)
    
    # make a grayscale copy of the template image, and store its width and height
    grayTemplateImage = cv2.cvtColor(colorTemplateImage, cv2.COLOR_BGR2GRAY)
    templateImageWidth = grayTemplateImage.shape[1]
    templateImageHeight = grayTemplateImage.shape[0]
    
    # scale our width and height according to our given image scale
    scaledWidth = int(templateImageWidth * imageScale)
    scaledHeight = int(templateImageHeight * imageScale)
    
    # resize our template and if needed, the mask 
    scaledGrayTemplateImage = cv2.resize(grayTemplateImage, (scaledWidth, scaledHeight))
    
    # get the results of our template match and extract the location of the best match from our results
    if maskImage is None:
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
    else:
        scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
    
    # extract our values from the template match result
    minVal, _, minLoc, _ = cv2.minMaxLoc(matchResult)
    
    # compute the area of our subimage in our source image
    x1 = minLoc[0]
    x2 = x1 + scaledWidth
    y1 = minLoc[1]
    y2 = y1 + scaledHeight
    
    # get our subimage
    subimage = colorSourceImage[y1:y2, x1:x2]

    # return the subimage along with some other values
    return subimage, scaledWidth, scaledHeight, minVal, matchResult


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
        
        minVal, _, minLoc, _ = cv2.minMaxLoc(matchResult)

        # check to see if our current match is better than our best match
        if minVal < bestMatch:
            # if so, update our tracking variables and reset our bad counter
            bestMatch = minVal
            bestMatchLocation = minLoc
            bestWidth = scaledWidth
            bestHeight = scaledHeight
            bestScale = scale
            bestMatchResult = matchResult
            badCounter = 0
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


# get the name of the student given the stats image
def getStudentName(statsImage, imageScale):
    # process and crop our image to the name subarea. also adds an additional "name" before the image to help with reading it
    processedNameSubimage = processNameImage(statsImage, imageScale)
    
    # read the image
    imageString = convertImageToString(processedNameSubimage)
    
    # the image should read: "Name: StudentNameHere", so we take out the "Name: "
    studentName = imageString.replace("Name: ", "")
    
    # return the name
    return studentName


# get the level from the stats image, can be used for both the Bond level and the Student level
def getStatsLevels(statsImage, imageScale, statsLevelsMaskImage, levelTemplateImages, levelMaskImages, matchThreshold, overlapThreshold):
    # crop out your target area from the stats image given a mask
    statsLevelsImage = cropImageWithMask(statsImage, statsLevelsMaskImage)
    
    # lists to keep track of our matches
    levelsWidths = []            # widths of the TM'ed level subimages
    levelsHeights = []           # heights of the TM'ed level subimages
    levelsNMSResults = []   # nms-filtered match results that passed our threshold
    levelsNMSLocations = [] # locations of said matches
    levelsNMSMatches = []   # the level template of the match (0-9)
    
    # go through range 0-9 and tm for the respective level template
    for level in range(10):
        # get the current level template and mask
        levelTemplateImage = levelTemplateImages[level]
        levelMaskImage = levelMaskImages[level]
        
        # TM for it in our levelsSubimage
        levelSubimage, levelSubimageWidth, levelSubimageHeight, levelSubimageResult, levelSubimageResults = subimageScaledSearch(statsLevelsImage, imageScale, levelTemplateImage, levelMaskImage)
        
        

##        #### TESTING PURPOSES TO DETERMINE BEST THRESHOLD ####
##        global MAX_HIT
##        global MIN_MISS
##        print(level, ":", levelSubimageResult)
##        if level in BOND_CHEAT_SHEET_ARRAY[COUNTER]:
##            if levelSubimageResult > MAX_HIT:
##                MAX_HIT = levelSubimageResult
##        else:
##            if levelSubimageResult < MIN_MISS:
##                MIN_MISS = levelSubimageResult
                
        
        # if the best result is below our threshold, filter our results and record them
        if levelSubimageResult < matchThreshold:
            # filter our results through NMS
            nmsResults, nmsLocations, levelCount, _ = filterResultsAndLocations(levelSubimageResults, levelSubimageWidth, levelSubimageHeight, matchThreshold, overlapThreshold)
            
            ## record our results
            # the reason why we extend widths, heights, and matches multiple times is because
            # results and locations may have multiple values. so we extend these ones by the
            # resultCount(levelCount) for when we start to furtther filter values from our arrays.
            levelsWidths.extend([levelSubimageWidth] * levelCount)
            levelsHeights.extend([levelSubimageHeight] * levelCount)
            levelsNMSResults.extend(nmsResults)
            levelsNMSLocations.extend(nmsLocations)
            levelsNMSMatches.extend([level] * levelCount)
    
    # convert our lists to np arrays for coding purposes
    levelsWidths = np.array(levelsWidths)
    levelsHeights = np.array(levelsHeights)
    levelsNMSResults = np.array(levelsNMSResults)
    levelsNMSLocations = np.array(levelsNMSLocations)
    levelsNMSMatches = np.array(levelsNMSMatches)
    
    # further filter our results. do so because sometimes numbers like "1" will TM to "4"s, and "5"s and "6"s, or "3"s and "8"s.
    # so with our array of all our matches thrown together, we filter it to make sure that we only have the best matches + no overlap.
    # a 3 should have a better match to 3 than 8 would to 3, so the 3 has priority. then we filter matches that have significatn overlap with the 3 (the 8)
    nms2Results, nms2Locations, levelCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
    
    # extract only our x coordinates from our locations array
    nms2XCoordinates = nms2Locations[:, 0]
    
    # nms2Order is an array of levelsNMSResults' indices of nms2Results values.
    # nms2Order is ordered in the order of nms2Results.
    # for example, if levelsNmsResults = [a, b, c, d, e, f] and nsm2Results = [f, b, d],
    # then nms2Order = [5, 1, 3].
    nms2Matches = levelsNMSMatches[nms2Order]
    
    # we order our nms2Matches by the x-coordinates, cause we read the level from left-to-right
    levelOrder = np.argsort(nms2XCoordinates)
    
    # put our level string together
    statsLevels = ""
    
    for levelIndex in levelOrder:
        statsLevels += str(nms2Matches[levelIndex])
    
    statsLevels = int(statsLevels)
    
    return statsLevels


# get the number of stars. can be used for student star or ue star
def getStarCount(sourceImage, imageScale, sourceStarMaskImage, starTemplateImage, starMaskImage, starMatchThreshold, starOverlapThreshold):
    # given the stats subimage, crop out the area where the stars should be using our mask
    sourceStarSubimage = cropImageWithMask(sourceImage, sourceStarMaskImage)
    
    # TM for star template in our starsSubimage
    _, starSubimageWidth, starSubimageHeight, starSubimageResult, starSubimageResults = subimageScaledSearch(sourceStarSubimage, imageScale, starTemplateImage, starMaskImage)
    
    # filter our results to be below the threshold and grab the coordinates of those matches
    nmsResults, nmsLocations, nmsStarCount, nmsOrder = filterResultsAndLocations(starSubimageResults, starSubimageWidth, starSubimageHeight, starMatchThreshold, starOverlapThreshold)
    
##    drawBoxes(sourceStarSubimage, starSubimageWidth, starSubimageHeight, nmsLocations)
    
    # return our star count
    return nmsStarCount


# get and return the info from our stats subimage. namely the student's name, bond, level, and star.
def getStudentStats(sourceImage, imageScale):
    # given our equipment image, TM for the stats with our scale 
    statsSubimage, _, _, statsSubimageResult, _ = subimageScaledSearch(sourceImage, imageScale, STATS_TEMPLATE_IMAGE, STATS_MASK_IMAGE)
    cv2.imshow("stats sub", statsSubimage)
    
##    studentName = getStudentName(statsSubimage, imageScale)
##    studentBond = getStatsLevels(statsSubimage, imageScale, STATS_BOND_MASK_IMAGE, BOND_TEMPLATE_IMAGES, BOND_MASK_IMAGES, BOND_MATCH_THRESHOLD, BOND_OVERLAP_THRESHOLD)
##    studentLevel = getStatsLevels(statsSubimage, imageScale, STATS_LEVEL_MASK_IMAGE, LEVEL_TEMPLATE_IMAGES, LEVEL_MASK_IMAGES, LEVEL_MATCH_THRESHOLD, LEVEL_OVERLAP_THRESHOLD)
    studentStar = getStarCount(statsSubimage, imageScale, STATS_STAR_MASK_IMAGE, STAR_TEMPLATE_IMAGE, STAR_MASK_IMAGE, STAR_MATCH_THRESHOLD, STAR_OVERLAP_THRESHOLD)
    
    global COUNTER
    
##    if BOND_CHEAT_SHEET[COUNTER] != studentBond:
##        print("cheat bond:", BOND_CHEAT_SHEET[COUNTER])
##        print("stud bond:", studentBond)
    
##    if LEVEL_CHEAT_SHEET[COUNTER] != studentLevel:
##        print("cheat level:", LEVEL_CHEAT_SHEET[COUNTER])
##        print("student level:", studentLevel)
    
##    if STAR_CHEAT_SHEET[COUNTER] != studentStar:
##        print("cheat star:", STAR_CHEAT_SHEET[COUNTER])
##        print("student star:", studentStar)
    
    
    
    return 1, 1, 1, 1
    return studentName, studentBond, studentLevel, studentStar


# checks if the source has the equipped template in it. returns true or false
def checkEquipped(sourceImage, imageScale, sourceEMaskImage, eTemplateImage, eMaskImage, eMatchThreshold):
    # given the source image, get the area of where the "E" should be.
    # the "E" represents whether there an item is equipped.
    gearESubimage = cropImageWithMask(sourceImage, sourceEMaskImage)
    
    # template match for the "E"
    eSubimage, _, _, eSubimageResult, _ = subimageScaledSearch(gearESubimage, imageScale, eTemplateImage, eMaskImage)
    
    # if the "E" match result is lower than our threshold, that means the "E" was found and there is a gear equipped
    if eSubimageResult < eMatchThreshold:
        return True
    else:
        return False


# given the gear image, returns the level of the tier of the gear
def getGearTier(gearImage, imageScale):
    # given the specific gear image (gear1, gear2, or gear3), we get the subimage of where the tier should be
    gearTierSubimage = cropImageWithMask(gearImage, GEAR_TIER_MASK_IMAGE)
    
    # variables to keep track of which tier template matches best with our image
    bestTierResult = float("inf")
    bestTier = 0
    
    # calculate the amount of available tiers
    tierCount = len(TIER_TEMPLATE_IMAGES)
    
    # go through all the tier templates and see which one matches best 
    for tier in range(tierCount):
        # get the current tier template image
        tierTemplateImage = TIER_TEMPLATE_IMAGES[tier]
        
        # template match the current tier template in our subimage
        _, _, _, tierSubimageResult, _ = subimageScaledSearch(gearTierSubimage, imageScale, tierTemplateImage, TIER_MASK_IMAGE)
        
        # check to see if our current match is better than our best match
        # update our variables if so
        if tierSubimageResult < bestTierResult:
            bestTierResult = tierSubimageResult
            bestTier = tier + 1 # +1 because counter starts at 0
    
    # return our best-matched tier
    return bestTier


# given the equipment subimage, return an array of the student's gears tiers
def getStudentGears(equipmentImage, imageScale, studentLevel):
    # given our equipment image, TM for the gears area
    gearsSubimage, _, _, gearsSubimageResult, _ = subimageScaledSearch(equipmentImage, imageScale, GEARS_TEMPLATE_IMAGE, GEARS_MASK_IMAGE)    
    
    # get the gear slot count
    gearCount = len(GEAR_SLOT_LEVEL_REQUIREMENTS)

    # array to keep track of our student's gear's tiers.
    # we premake it because even if student doesnt meet the level req,
    # that means the tier is 0, or in other words nothing is equipped
    studentGears = np.zeros(gearCount, np.int)
    
    # go through all of our individual gears
    for gear in range(gearCount):
        # get the current gear's level req
        gearSlotLevelRequirement = GEAR_SLOT_LEVEL_REQUIREMENTS[gear]
        
        # if student level isn't high enough, we break
        if studentLevel < gearSlotLevelRequirement:
            break
        
        # get the respective gear mask
        gearMaskImage = GEAR_MASK_IMAGES[gear]
        
        # get the gear subimage
        gearSubimage = cropImageWithMask(gearsSubimage, gearMaskImage)
        
        # check if the current gear slot has anything equipped
        isGearEquipped = checkEquipped(gearSubimage, imageScale, GEAR_E_MASK_IMAGE, E_TEMPLATE_IMAGE, E_MASK_IMAGE, E_MATCH_THRESHOLD)
        
        # if a gear is equipped, we get the tier
        # else we do nothing, leaving it at 0, representing no equip
        if isGearEquipped:
            # get the tier of the image and add it to our list
            gearTier = getGearTier(gearSubimage, imageScale)
            studentGears[gear] = gearTier
    
    # return our array of gears
    return studentGears


#
def getStudentUE(equipmentImage, imageScale, studentStar):
    ueSubimage, _, _, ueSubimageResult, _ = subimageScaledSearch(equipmentImage, imageScale, UE_TEMPLATE_IMAGE, UE_MASK_IMAGE)    
    
    # initialize our variables
    ueStar = 0
    ueLevel = 0
    
    # check to make sure the student's star is high enough to even own a UE
    if studentStar >= UE_SLOT_STAR_REQUIREMENT:
        isUEEquipped = checkEquipped(ueSubimage, imageScale, UE_E_MASK_IMAGE, E_UE_TEMPLATE_IMAGE, E_UE_MASK_IMAGE, E_UE_MATCH_THRESHOLD)

        if isUEEquipped:
            ueStar = getStarCount(ueSubimage, imageScale, UE_STAR_MASK_IMAGE, STAR_UE_TEMPLATE_IMAGE, STAR_UE_MASK_IMAGE, STAR_UE_MATCH_THRESHOLD, STAR_UE_OVERLAP_THRESHOLD)
##            ueLevel = getUELevel()
        
    # return our ue's star and level
    return ueStar, ueLevel



directory = "student example"
scale = 1.0
for fileName in os.listdir(directory):
    f = os.path.join(directory, fileName)

    sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)

    scale = SCALE_CHEAT_SHEET[COUNTER]
    studentStar = STAR_CHEAT_SHEET[COUNTER]
    studentLevel = LEVEL_CHEAT_SHEET[COUNTER]
        
    print(f, scale)
    
##    studentName, studentBond, studentLevel, studentStar = getStudentStats(sourceImage, scale)
    equipmentImage, _, _, _, _ = subimageScaledSearch(sourceImage, scale, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
    studentStar = STAR_CHEAT_SHEET[COUNTER]
    studentUE = getStudentUE(equipmentImage, scale, studentStar)
    
    COUNTER += 1
