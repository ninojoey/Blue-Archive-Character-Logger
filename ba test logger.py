import cv2
import time
import random
import numpy as np
from pytesseract import pytesseract


SITE_VERSION = "1.3.5"
SCALE_INCREMENT = 0.01
# cv2.TM_CCOEFF_NORMED returns inf
# ccorr is bad because "we're dealing with discrete digital signals that have a
# defined maximum value (images), that means that a bright white patch of the image will basically
# always have the maximum correlation" and ccoeff pretty much fixes that problem
# sqdiff calculates the intensity in DIFFERENCE of the images. so you wanna look for the min if using sqdiff
# you want to use NORMED functions when template matching with templates of different sizes
#TEMPLATE_MATCH_METHODS_ARRAY = [cv2.TM_CCOEFF]#, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TEMPLATE_MATCH_METHOD = cv2.TM_SQDIFF_NORMED
PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GEAR_X_OFFSET = 4
GEAR_Y_OFFSET = 6
BAD_COUNTER_MAX = 150000
OVERLAP_THRESHOLD = 0.35
TIME = time.time()

# indices
# [2:] = starting with index 2
# [:2] up to index 2
# [1:2] items from index 1 to 2
# [::2] = every 2 items, in other words skip 1
# [::-1] = every item, back in backwards order
# [2::2] starting with index 2, then every other item afterwards
# [:8:2] ending at index 8, get every other element starting from the beg
# [1:8:2] starting at index 1 and ending at index 8, get every other item
# [:,:] all elements in all rows. comma is normal.


# THINGS LEARNED AND USED
# template matching
#    multi scale matching
#    multi object matching
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


# logic for pulling from json
# take in import txt. check export and site version. check if it lines up with our
# copy of the database (variables inside database will be used to check)
# if lines up, continue with our database. if not, we update our database by
# running the json grabber thingie and write to the database.




def convertImageToString(img):
    pytesseract.tesseract_cmd = PATH_TO_TESSERACT
    imageToString = pytesseract.image_to_string(img)
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
    imageChannelCount = image.shape[2]
    if imageChannelCount == 3:
        return None
    
    
    ## else we make a mask based on the alpha channel.
    ## we'll scale the pixels according to the transparency
    # get the dimensions of the image
    imageWidth = image.shape[1]
    imageHeight = image.shape[0]
    
    # create a completely black image of the same dimensions
    maskImage = np.ones((imageHeight, imageWidth), np.uint8) * 255
    
    # turn the img's transparency array into a scalar array
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
def processImage(colorImage, maskImage = None):
    # convert the image to gray
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("1", grayImage)
    
    
    # image threshold, basically turns it black and white,
    # highlighting important textures and features
    processedImage = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #cv2.imshow("2", processedImage)
    

    # floodfill our image to black out the background
    startPoint = (0, 0)         # start at the first pixel
    newColor = (0, 0, 0, 255)   # color to flood with, black in this case
    cv2.floodFill(processedImage, None, startPoint, newColor)
    #cv2.imshow("3", processedImage)


##    # morph open our image to remove general noise
##    kernel = np.ones((1,1),np.uint8)
##    processedImage = cv2.morphologyEx(processedImage, cv2.MORPH_OPEN, kernel, iterations=1)
##    #cv2.imshow("4", processedImage)
    
    
##    # find contours and remove small noise
##    contours = cv2.findContours(processedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##    contours = contours[0] if len(contours) == 2 else contours[1]
##    for contour in contours:
##        area = cv2.contourArea(contour)
##        if area < 20:
##            cv2.drawContours(processedImage, [contour], -1, 0, -1)
##
##    #cv2.imshow("5", processedImage)

    
    # apply mask if there is one 
    if maskImage is not None:
        processedImage = cv2.bitwise_and(processedImage, maskImage)
        #cv2.imshow("6", processedImage)

    processedImage = cv2.cvtColor(processedImage, cv2.COLOR_GRAY2BGR)

    return processedImage


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
def nms(templateMatchMethod, matchResults, matchLocations, matchWidth, matchHeight):
    # unpack our arrays
    x1MatchCoordinates = matchLocations[1]
    x2MatchCoordinates = x1MatchCoordinates + matchWidth
    y1MatchCoordinates = matchLocations[0]
    y2MatchCoordinates = y1MatchCoordinates + matchHeight
    
    # determine which order to run through the results/coordinates.
    # we do it based on the results, starting with the highest value.
    # but since np.argsort sorts from low to high, we reverse it with [::-1].
    # we create a dedicated array instead of just sorting the results array itself
    # since we're working with other arrays (the boxes array which has x and y coordinates)
    if templateMatchMethod == cv2.TM_SQDIFF or templateMatchMethod == cv2.TM_SQDIFF_NORMED:
        indexOrder = np.argsort(matchResults)[::1]
    else:
        indexOrder = np.argsort(matchResults)[::-1]
    
    # get the area of the box. they should all be the same area
    matchArea = matchWidth * matchHeight

    # create an array to store our overlap percentages
    overlap = []

    # array to store the coordinates of our best matches, filtering out overlappers
    bestMatchLocations = []
    

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

        ## determine the overlap of the other matches with the current match.
        ## to find the overlapping area, the x and y values should be furthest away
        ## from the edges of the original. eg if comparing our x1 (top left) coordinate,
        ## compare to coordinate furthest from top left.
        # go through respective coordinates (x1, x2, y1, y2) and compare the
        # best match coordinate with its respective matchCoordinates array.
        x1OverlapCoordinates = np.maximum(x1BestMatchCoordinate, x1MatchCoordinates)
        x2OverlapCoordinates = np.minimum(x2BestMatchCoordinate, x2MatchCoordinates)
        y1OverlapCoordinates = np.maximum(y1BestMatchCoordinate, y1MatchCoordinates)
        y2OverlapCoordinates = np.minimum(y2BestMatchCoordinate, y2MatchCoordinates)
        
        ## calculate the area of overlap
        # we do a max of 0, because if you have a negative edges, that means
        # there isn"t any overlap, and if you have 2 negative edges, the area will
        # still be positive, resulting in a false positive.
        overlapWidths  = np.maximum(0, x2OverlapCoordinates - x1OverlapCoordinates)
        overlapHeights = np.maximum(0, y2OverlapCoordinates - y1OverlapCoordinates)
        overlapAreas = overlapWidths * overlapHeights

        # calculate the percentage of overlap with our matched area
        overlapPercentages = overlapAreas / matchArea

        # delete the boxes where the overlap is over a certain threshold.
        # in this case, we delete the entries of the indicies so we don't
        # iterate over them later. also delete the one we just worked on cause it's done
        
        indexDelete = np.where(overlapPercentages > OVERLAP_THRESHOLD)[0]
        if bestMatchIndex not in indexDelete:
            indexDelete = np.append(indexDelete, bestMatchIndex)
        
        indexOrder = np.setdiff1d(indexOrder, indexDelete, True)
        

    # return the coordinates of the filtered boxes
    return bestMatchLocations


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

    # if there is no mask, create one
    if maskImage is None:
        maskImage = createMaskFromTransparency(colorTemplateImage)
    
    # calculate the scaled width and height of our tempalte and mask
    scaledWidth  = int(templateImageWidth * scale)
    scaledHeight = int(templateImageHeight * scale)
    
    # resize our template and mask
    scaledGrayTemplateImage = cv2.resize(grayTemplateImage, (scaledWidth, scaledHeight))
    
    # get the results of our template match and extract the location of the best match from our results
    if maskImage is None:
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
    else:
        scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
        matchResult = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
##        matchResult, method = runAllTemplateMatchingMethods(graySourceImage, scaledGrayTemplateImage, scaledMaskImage)
    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResult)

    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        # calculate the area of our subimage in our source image
        x1 = minLoc[0]
        x2 = x1 + scaledWidth
        y1 = minLoc[1]
        y2 = y1 + scaledHeight
    else:
        # calculate the area of our subimage in our source image
        x1 = maxLoc[0]
        x2 = x1 + scaledWidth
        y1 = maxLoc[1]
        y2 = y1 + scaledHeight

    # get our subimage
    subimage = colorSourceImage[y1:y2, x1:x2]
    #print("best method:", method)

    # return the subimage along with some other values
    return subimage, scaledWidth, scaledHeight, matchResult


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
    
    # if there is no mask, create one
    if maskImage is None:
        maskImage = createMaskFromTransparency(colorTemplateImage)
    

    
    # variable to scale the template by
    scale = 1.0

    # variables that keep track of values of our best match
    if TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF or TEMPLATE_MATCH_METHOD == cv2.TM_SQDIFF_NORMED:
        bestMatch = float("inf")
    else:
        bestMatch = 0
    bestMatchLocation = (0, 0)
    bestWidth = templateImageWidth
    bestHeight = templateImageHeight
    bestScale = scale
    bestMatchResult = []
    bestMethod = ""

    # values of our newly scaled template/image
    scaledWidth  = templateImageWidth
    scaledHeight = templateImageHeight

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




EQUIPMENT_TEMPLATE_IMG_PATH = "skill template.png"
EQUIPMENT_MASK_IMAGE_PATH = "skill mask.png"
STATS_TEMPLATE_IMG_PATH = "stats template.png"
STATS_MASK_IMAGE_PATH = "stats mask.png"

STARS_MASK_IMAGE_PATH = "stats star mask.png"
HEART_MASK_IMAGE_PATH = "stats heart mask.png"
LEVEL_MASK_IMAGE_PATH = "stats level mask.png"
NAME_MASK_IMAGE_PATH  = "stats name mask.png"


SOURCE_IMG_PATH = "hibiki ipad.png"
EQUIPMENT_SUBIMG_PATH = "hibiki skill.png"
STATS_SUBIMG_PATH = "hibiki stat.png"


def main():
    # load our original image
    sourceImage = cv2.imread(SOURCE_IMG_PATH, -1)
    
    # load our window skill window template along with its mask
    equipmentTemplateImage = cv2.imread(EQUIPMENT_TEMPLATE_IMG_PATH, -1)
    equipmentMaskImage =  cv2.imread(EQUIPMENT_TEMPLATE_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    
    # multi-scale template match for our skill template
    equipmentSubimage, equipmentSubimageWidth, equipmentSubimageHeight, equipmentSubimageScale, equipmentSubimageMatchResult = \
                       subimageMultiScaleSearch(sourceImage, equipmentTemplateImage, equipmentMaskImage)
    cv2.imshow("equipmentSubimage", equipmentSubimage)
    cv2.imwrite(EQUIPMENT_SUBIMG_PATH, equipmentSubimage)

    
    # load in the stats template image and mask
    statsTemplateImage = cv2.imread(STATS_TEMPLATE_IMG_PATH, -1)
    statsMaskImage = cv2.imread(STATS_MASK_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

    # scaled tempalte match for our stat window
    statsSubimage,_,_,_ = subimageScaledSearch(sourceImage, statsTemplateImage, statsMaskImage, equipmentSubimageScale)
    cv2.imshow("statsSubimage", statsSubimage)
    cv2.imwrite(STATS_SUBIMG_PATH, statsSubimage)
    
    
    
    ### HERE WE SHOULD EXTRACT INFO FROM THE STATS WINDOW ###
    # name, level, bond, star = extractStatsInfo(statsSubimage)


    ### NEXT WE WORK ON OUR SKILL WINDOW
    # ex, basic, passive, sub = extractSkillInfo(equipmentSubimage, star)
    # ue, ue_level = extractUEInfo(equipmentSubimage, star)
    # gear1, gear2, gear3 = extractGearInfo(equipmentSubimage, level)


    

    # since one of the objects in the skill window is an image and not text, i"ll load in a copy of said object
    maxSkillTemplate_rgb = cv2.imread("max template.png", -1)
    cv2.imshow("max maxSkillTemplate_rgb", maxSkillTemplate_rgb)
    #max_subimage = findSubimageInImage(subimage, max_template)
    #printAndShowErrDiff(max_template, max_subimage)

    # get the result of our template search
    maxSkillSubimage_rgb, matchWidth, matchHeight, matchResults = subimageRGBSearch(skillWindowSubimage_rgb, maxSkillTemplate_rgb)
    cv2.imshow("maxSkillSubimage_rgb", maxSkillSubimage_rgb)

    # match_result is a 2d array where each value is the tempalte"s match percentage to
    # the original image at that given coordinate on the image. so the result at [0][0]
    # corresponds to the template"s match percentage at (0,0) on the image.
    # sift through our result and get the ones that are above our threshold
    bestMatchLocations = np.where(matchResults >= 0.80)
    filteredMatchesResults = np.extract(matchResults >= 0.80, matchResults)

    # run our coordinates and results through NMS. now we should have coordinates of the 
    # best results with little-to-no overlapping areas
    nmsCoordinates = nms(filteredMatchesResults, bestMatchLocations, matchWidth, matchHeight)


    # now with nms done, we have to determine where those max"s belong to and where levels would belong to



    #gearWindowTemplate_rgb = cv2.imread("T6_Hairpin.png", cv2.IMREAD_UNCHANGED)
    #gearWindowgrayTemplateImage = cv2.cvtColor(gearWindowTemplate_rgb, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gearWindowgrayTemplateImage", gearWindowgrayTemplateImage)
    #gearWindowSubimage_rgb,_,_,_ = subimageMultiScaleSearch(statWindowSubimage_rgb, gearWindowTemplate_rgb)
    #cv2.imshow("gearWindowSubimage_rgb", gearWindowSubimage_rgb)









    

### import images
##s_img = cv2.imread("T5_Necklace.png", -1)
##card_img = cv2.imread("Card_Bg.png", -1)
##l_imga = cv2.resize(card_img, (312, 256))
##
##
### combine our images
##l_img = combineTransparentImages(l_imga, s_img, GEAR_X_OFFSET, GEAR_Y_OFFSET)
###cv2.imshow("l_img", l_img)
##
##
### read in our mask. i made this to block out the "E", "TX", and "Lv.XX"
##maskImage = cv2.imread("gear mask.png", cv2.IMREAD_GRAYSCALE) 
##maskImage = cv2.resize(maskImage, (312, 256))
##
##
### read in our source image
##akashoes = cv2.imread("akane gear.png", cv2.IMREAD_COLOR)
##
##startTime = time.time()
### template match for our combbined image with our mask
##gearSub, bestWidth, bestHeight, bestScale, bestResult = subimageMultiScaleSearch(akashoes, l_img, maskImage)
##gearTime = startTime - time.time()
##cv2.imshow("gearSub", gearSub)
##
##
### read in our mask and apply to our flooded image
##levelMask = cv2.imread("level only mask.png", cv2.IMREAD_GRAYSCALE)
##levelMask = cv2.resize(levelMask, (bestWidth, bestHeight))
##
##processedImage = processImage(gearSub, levelMask)
##
### read the image
##convertImageToString(processedImage)




##
##source = cv2.imread("maki.png", cv2.IMREAD_COLOR)
##skillTemplate = cv2.imread("skill template.png", cv2.IMREAD_COLOR)
##skillMask = cv2.imread("skill mask.png", cv2.IMREAD_GRAYSCALE)
##
##startTime = time.time()
##skillSubimage, bestWidth, bestHeight, bestScale, bestResult = subimageMultiScaleSearch(source, skillTemplate, skillMask)
##skillTime = startTime - time.time()
##cv2.imshow("skillSubimage", skillSubimage)
##
##
##statsTemplate = cv2.imread("stats template.png", cv2.IMREAD_COLOR)
##statsMask = cv2.imread("stats mask.png", cv2.IMREAD_GRAYSCALE)
##
##
##startTime = time.time()
##statsSubimage, bestWidth, bestHeight, bestResult = subimageScaledSearch(source, statsTemplate, statsMask, bestScale)
##statTime = startTime - time.time()
##print("best reuslt", bestResult)
##cv2.imshow("statsSubimage", statsSubimage)
##cv2.imwrite("maki stats.png", statsSubimage)
####
####print(gearTime)
##print(skillTime)
##print(statTime)
##
##
##hibikiStats = cv2.imread("maki stats.png", cv2.IMREAD_COLOR)
##processedHibiki = processImage(hibikiStats)
##
##starsMask = cv2.imread("stats star mask.png", cv2.IMREAD_GRAYSCALE)
##heartMask = cv2.imread("stats heart mask.png", cv2.IMREAD_GRAYSCALE)
##levelMask = cv2.imread("stats level mask.png", cv2.IMREAD_GRAYSCALE)
##nameMask  = cv2.imread("stats name mask.png", cv2.IMREAD_GRAYSCALE)
##
##
##starsMask = resizeImage(starsMask, processedHibiki)
##heartMask = resizeImage(heartMask, processedHibiki)
##levelMask = resizeImage(levelMask, processedHibiki)
##nameMask = resizeImage(nameMask, processedHibiki)
##
##hibikiStars = cv2.bitwise_and(processedHibiki, processedHibiki, mask = starsMask)
##hibikiHeart = cv2.bitwise_and(processedHibiki, processedHibiki, mask = heartMask)
##hibikiLevel = cv2.bitwise_and(processedHibiki, processedHibiki, mask = levelMask)
##hibikiName  = cv2.bitwise_and(processedHibiki, processedHibiki, mask = nameMask)
##
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



####src1_mask=cv2.cvtColor(src1_mask,cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
####mask_out=cv2.subtract(src1_mask,src1)
####mask_out=cv2.subtract(src1_mask,mask_out)
####cv2.imshow("maskout", mask_out)
##
##
##
##sourceImage = cv2.imread("maki beeg.png", cv2.IMREAD_COLOR)
##equipmentTemplateImage = cv2.imread("skill template.png", cv2.IMREAD_COLOR)
##equipmentMaskImage = cv2.imread("skill mask.png", cv2.IMREAD_GRAYSCALE)
##equipmentSubimage, _, _, equipmentSubimageScale, _ = subimageMultiScaleSearch(sourceImage, equipmentTemplateImage, equipmentMaskImage)
##
##ueTemplateImage = cv2.imread("ue template.png", cv2.IMREAD_COLOR)
##ueMaskImage = cv2.imread("ue mask.png", cv2.IMREAD_GRAYSCALE)
##ueSubimage, _, _, _ = subimageScaledSearch(equipmentSubimage, ueTemplateImage, ueMaskImage, equipmentSubimageScale)
##
##cv2.imshow("ueSubimage", ueSubimage)
##print("1")
##
##ueStarTemplateImage = cv2.imread("ue star template.png", -1)
##ueStarMaskImage = createMaskFromTransparency(ueStarTemplateImage)
##ueStarMasked = cv2.bitwise_and(ueStarTemplateImage, ueStarTemplateImage, mask = ueStarMaskImage)
##cv2.imshow("ueStarMasked", ueStarMasked)
##ueStarSubimage, ueStarWidth, ueStarHeight, ueStarScale, ueStarMatchResults = subimageMultiScaleSearch(ueSubimage, ueStarTemplateImage, ueStarMaskImage)
##cv2.imshow("ueStarTemplateImage", ueStarTemplateImage)
##cv2.imshow("ueStarMaskImage", ueStarMaskImage)
##cv2.imshow("ueStarSubimage", ueStarSubimage)
##
##print("2")
##
##bestUEStarMatchLocations = np.where(ueStarMatchResults >= 0.99)
##filteredUEStarMatchesResults = np.extract(ueStarMatchResults >= 0.99, ueStarMatchResults)
##print ( len ( filteredUEStarMatchesResults))
##print("3")
##### run our coordinates and results through NMS. now we should have coordinates of the 
##### best results with little-to-no overlapping areas
####nmsCoordinates = nms(filteredUEStarMatchesResults, bestUEStarMatchLocations, ueStarWidth, ueStarHeight)
####print("4")







##
##sourceImage = cv2.imread("airi.png", cv2.IMREAD_COLOR)
##equipmentTemplateImage = cv2.imread("skill template.png", cv2.IMREAD_COLOR)
##equipmentMaskImage = cv2.imread("skill mask.png", cv2.IMREAD_GRAYSCALE)
##
##equipmentSubimage, _, _, equipmentSubimageScale, _ = subimageMultiScaleSearch(sourceImage, equipmentTemplateImage, equipmentMaskImage)
##cv2.imshow("equipmentSubimage", equipmentSubimage)
##
##ueTemplateImage = cv2.imread("ue template.png", cv2.IMREAD_COLOR)
##ueMaskImage = cv2.imread("ue mask.png", cv2.IMREAD_GRAYSCALE)
##ueSubimage, _, _, _ = subimageScaledSearch(equipmentSubimage, ueTemplateImage, ueMaskImage, equipmentSubimageScale)
##cv2.imshow("ueSubimage", ueSubimage)
##
##
##ueStarsMaskImage = cv2.imread("ue stars mask.png", cv2.IMREAD_GRAYSCALE)
##ueStarsSubimage = cropImageWithMask(ueSubimage, ueStarsMaskImage)
##cv2.imwrite("airi ue stars subimage.png", ueStarsSubimage)
##cv2.imshow("ueStarsSubimage", ueStarsSubimage)
##


ueStarsSubimage = cv2.imread("maki ue stars subimage.png", cv2.IMREAD_COLOR)
ueStarTemplate = cv2.imread("ue star template.png", cv2.IMREAD_COLOR)
ueStarMask = cv2.imread("ue star mask.png", cv2.IMREAD_GRAYSCALE)
##ueStarMask = createMaskFromTransparency(ueStarTemplate)
ueStarSubimage, matchWidth, matchHeight, ueStarScale, ueStarResult = subimageMultiScaleSearch(ueStarsSubimage, ueStarTemplate, ueStarMask)#, equipmentSubimageScale)

cv2.imshow("ueStarSubimage", ueStarSubimage)
ueStarSubimage = cv2.resize(ueStarSubimage, (ueStarSubimage.shape[1] *5,ueStarSubimage.shape[0] *5))
cv2.imshow("resueStarSubimage", ueStarSubimage)

bestMatchLocations = np.where(ueStarResult <= 0.2)
filteredMatchesResults = np.extract(ueStarResult <= 0.2, ueStarResult)

# run our coordinates and results through NMS. now we should have coordinates of the 
# best results with little-to-no overlapping areas
nmsCoordinates = nms(TEMPLATE_MATCH_METHOD, filteredMatchesResults, bestMatchLocations, matchWidth, matchHeight)

for index in range(len(nmsCoordinates)):
    color = (0, index/len(nmsCoordinates) * 255, 0)
    cv2.rectangle(ueStarsSubimage, nmsCoordinates[index], (nmsCoordinates[index][0] + matchWidth, nmsCoordinates[index][1] + matchHeight), color, 1)

cv2.imshow("ueStarsSubimage", ueStarsSubimage)
ueStarsSubimage = cv2.resize(ueStarsSubimage, (ueStarsSubimage.shape[1] *5,ueStarsSubimage.shape[0] *5))
cv2.imshow("resueStarsSubimage", ueStarsSubimage)
#cv2.imwrite("airi ue stars boxed subimage.png", ueStarsSubimage)
print(len(nmsCoordinates))
