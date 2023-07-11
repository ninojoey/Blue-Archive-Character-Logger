import os
import Constants
import cv2
import time
import random
import numpy as np
from pytesseract import pytesseract
from Constants import *


MAX_HIT = 0
MIN_MISS = float("inf")

COUNTER = 0


print (SCALE_CHEAT_SHEET)

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
    contours = cv2.findContours(processedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cv2.drawContours(processedImage, contours, -1, 0, 1)
    
    blankImage1 = np.ones((processedImage.shape[0], processedImage.shape[1]), np.uint8) * 255
    cv2.drawContours(blankImage1, contours, -1, 0, cv2.FILLED)
##    cv2.imshow("RETR_LIST", processedImage)
    cv2.imshow("b RETR_LIST", blankImage1)
    

##    # morph open our image to remove general noise
##    kernel = np.ones((1, 1),np.uint8)
##    
##    erosion = cv2.erode(processedImage, kernel, iterations = 1)
##    cv2.imshow("erode", erosion)
##    
##    dilation = cv2.dilate(processedImage, kernel,iterations = 1)
##    cv2.imshow("dilation", dilation)
##
##    # or cv2.MORPH_OPEN
##    opening = cv2.morphologyEx(processedImage, cv2.MORPH_OPEN, kernel)
##    cv2.imshow("opening", opening)
    
    # floodfill our image to black out the background
    startPoint = (10, 10)         # start at the first pixel
    newColor = (0, 0, 0, 255)   # color to flood with, black in this case
    cv2.floodFill(processedImage, None, startPoint, newColor)
    
    
    # apply mask if there is one 
    if maskImage is not None:
        maskImage = resizeImage(maskImage, processedImage)
        processedImage = cv2.bitwise_and(processedImage, maskImage)

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
    if type(matchWidth) == int:
        matchWidth = np.full_like(matchResults, matchWidth)

    if type(matchHeight) == int:
        matchHeight = np.full_like(matchResults, matchHeight)

    
    
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

##    #### GET MAX AND MIN OVERLAP ####
##    x1MatchCoordinates = nmsLocations[:,0]
##    x2MatchCoordinates = x1MatchCoordinates + matchWidth[nmsOrder]
##    y1MatchCoordinates = nmsLocations[:,1]
##    y2MatchCoordinates = y1MatchCoordinates + matchHeight[nmsOrder]
##    allOverlapPercentages = []
##    
##    for index in range(nmsCount):
##
##        x1BestMatchCoordinate = x1MatchCoordinates[index]
##        x2BestMatchCoordinate = x2MatchCoordinates[index]
##        y1BestMatchCoordinate = y1MatchCoordinates[index]
##        y2BestMatchCoordinate = y2MatchCoordinates[index]
##        
##        # get its area
##        bestMatchWidth = x2BestMatchCoordinate - x1BestMatchCoordinate
##        bestMatchHeight = y2BestMatchCoordinate - y1BestMatchCoordinate
##        bestMatchArea = bestMatchWidth * bestMatchHeight
##        
##        x1OverlapCoordinates = np.maximum(x1BestMatchCoordinate, x1MatchCoordinates)
##        x2OverlapCoordinates = np.minimum(x2BestMatchCoordinate, x2MatchCoordinates)
##        y1OverlapCoordinates = np.maximum(y1BestMatchCoordinate, y1MatchCoordinates)
##        y2OverlapCoordinates = np.minimum(y2BestMatchCoordinate, y2MatchCoordinates)
##        
##        ## calculate the area of overlap
##        # we do a max of 0, because if you have a negative edges, that means there isn't 
##        # any overlap and actually represents the area of empty space between the two matches.
##        # and if you have 2 negative edges, the area will
##        # still be positive, resulting in a false positive.
##        overlapWidths  = np.maximum(0, x2OverlapCoordinates - x1OverlapCoordinates)
##        overlapHeights = np.maximum(0, y2OverlapCoordinates - y1OverlapCoordinates)
##        overlapAreas = overlapWidths * overlapHeights
##
##        overlapPercentages = overlapAreas / bestMatchArea
##
##        allOverlapPercentages.append(overlapPercentages)
##        
##        # here i wanna go through the matrix. change max_hit from the area within row AND col < star_cheat[counter]
##        print(overlapPercentages)
##    
##    allOverlapPercentages = np.array(allOverlapPercentages)
##    allOverlapPercentages = np.reshape(allOverlapPercentages, (nmsCount, nmsCount))
##    
##    starCount = STAR_CHEAT_SHEET[COUNTER]
##    print("star:", starCount)
##
##    overlapHits = allOverlapPercentages[:starCount, :starCount]
##    overlapHits = overlapHits.flatten()
##
##    overlapMisses = np.setdiff1d(allOverlapPercentages, overlapHits)
##    print("hits:", overlapHits)
##    print("miss:", overlapMisses)
##    overlapHits = np.setdiff1d(overlapHits, [1.0])
##    overlapMisses = np.setdiff1d(overlapMisses, [0.0])
##    if len(overlapHits) > 0:
##        if max(overlapHits) > MAX_HIT:
##            MAX_HIT = max(overlapHits)
##    if len(overlapMisses) > 0:
##        if min(overlapMisses) < MIN_MISS:
##            MIN_MISS = min(overlapMisses)
##
##    #######################################################


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


def drawBoxes(sourceImage, nmsLocations, templateWidth, templateHeight):
    x2 = nmsLocations[:,0] + templateWidth
    y2 = nmsLocations[:,1] + templateHeight
    
    for index in range(len(nmsLocations)):
        colorr = np.array([0, 0, 255]) * index / len(nmsLocations)
        location = nmsLocations[index]
        cv2.rectangle(sourceImage, (location), (x2[index], y2[index]), colorr, 1)

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
    bestMatch = float("inf")
    

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
    # so we scale the template down to match the source"
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
    
    return subimage, bestWidth, bestHeight, bestScale, bestMatchResult


# get the name of the student given the stats image
def getStudentName(statsImage, imageScale):
    # process and crop our image to the name subarea. also adds an additional "name" before the image to help with reading it
    processedNameSubimage = processNameImage(statsImage, imageScale)
    
    # read the image
    imageString = convertImageToString(processedNameSubimage)
    
    # the image should read: "Name: StudentNameHere", so we take out the "Name: "
    studentName = imageString.replace("Name: ", "")
    studentName = studentName.strip()
    
    # return the name
    return studentName


# get the level from the stats image, can be used for both the Bond level and the Student level
def getLevels(sourceImage, imageScale, sourceLevelsMaskImage, levelTemplateImages, levelMaskImages, matchThreshold, overlapThreshold):
    # crop out your target area from the stats image given a mask
    sourceLevelsImage = cropImageWithMask(sourceImage, sourceLevelsMaskImage)
    
    # lists to keep track of our matches
    levelsWidths = []            # widths of the TM'ed level subimages
    levelsHeights = []           # heights of the TM'ed level subimages
    levelsNMSResults = []   # nms-filtered match results that passed our threshold
    levelsNMSLocations = [] # locations of said matches
    levelsNMSMatches = []   # the level template of the match (0-9)
    
    # go through range 0-9 and tm for the respective level template
    levelCount = len(levelTemplateImages)
    for level in range(levelCount):
        # get the current level template and mask
        levelTemplateImage = levelTemplateImages[level]
        levelMaskImage = levelMaskImages[level]
        
        # TM for it in our levelsSubimage
        levelSubimage, levelSubimageWidth, levelSubimageHeight, levelSubimageResult, levelSubimageResults = subimageScaledSearch(sourceLevelsImage, imageScale, levelTemplateImage, levelMaskImage)
        
##        #### TESTING PURPOSES TO DETERMINE BEST THRESHOLD ####
##        global MAX_HIT
##        global MIN_MISS
##        print(level, ":", levelSubimageResult)
##        if level in UE_LEVEL_CHEAT_SHEAT_ARRAY[COUNTER]:
##            if levelSubimageResult > MAX_HIT:
##                MAX_HIT = levelSubimageResult
##        else:
##            if levelSubimageResult < MIN_MISS:
##                MIN_MISS = levelSubimageResult
##        #######################################################
                
        
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
    
##    drawBoxes(sourceLevelsImage, nms2Locations, levelsWidths[nms2Order], levelsHeights[nms2Order])
    
    
    return statsLevels


# get the number of stars. can be used for student star or ue star
def getStarCount(sourceImage, imageScale, sourceStarMaskImage, starTemplateImage, starMaskImage, starMatchThreshold, starOverlapThreshold):
    # given the stats subimage, crop out the area where the stars should be using our mask
    sourceStarSubimage = cropImageWithMask(sourceImage, sourceStarMaskImage)
    
    # TM for star template in our starsSubimage
    _, starSubimageWidth, starSubimageHeight, starSubimageResult, starSubimageResults = subimageScaledSearch(sourceStarSubimage, imageScale, starTemplateImage, starMaskImage)
    
    # filter our results to be below the threshold and grab the coordinates of those matches
    nmsResults, nmsLocations, nmsStarCount, nmsOrder = filterResultsAndLocations(starSubimageResults, starSubimageWidth, starSubimageHeight, starMatchThreshold, starOverlapThreshold)
    
##    drawBoxes(sourceStarSubimage, starSubimageWidth, starSubimageHeight)
    
    # return our star count
    return nmsStarCount


# get and return the info from our stats subimage. namely the student's name, bond, level, and star.
def getStudentStats(sourceImage, imageScale):
    # given our equipment image, TM for the stats with our scale 
    statsSubimage, _, _, statsSubimageResult, _ = subimageScaledSearch(sourceImage, imageScale, STATS_TEMPLATE_IMAGE, STATS_MASK_IMAGE)
    cv2.imshow("stats sub", statsSubimage)
    
    studentName = getStudentName(statsSubimage, imageScale)
    studentBond = getLevels(statsSubimage, imageScale, STATS_BOND_MASK_IMAGE, BOND_LEVEL_TEMPLATE_IMAGES, BOND_LEVEL_MASK_IMAGES, BOND_LEVEL_MATCH_THRESHOLD, BOND_LEVEL_OVERLAP_THRESHOLD)
    studentLevel = getLevels(statsSubimage, imageScale, STATS_LEVEL_MASK_IMAGE, STATS_LEVEL_TEMPLATE_IMAGES, STATS_LEVEL_MASK_IMAGES, STATS_LEVEL_MATCH_THRESHOLD, STATS_LEVEL_OVERLAP_THRESHOLD)
    studentStar = getStarCount(statsSubimage, imageScale, STATS_STAR_MASK_IMAGE, STAR_TEMPLATE_IMAGE, STAR_MASK_IMAGE, STAR_MATCH_THRESHOLD, STAR_OVERLAP_THRESHOLD)
    
    return studentName, studentBond, studentLevel, studentStar


#
def getStudentSkills(equipmentImage, imageScale, studentStar):
    # grab skills mask depending on the students star. changes because some skills aren't
    # unlocked at lower stars
    skillsMaskImage = SKILLS_MASK_IMAGES[(studentStar-1)]
    
    # TM for the skills with our mask
    skillsSubimage, _, _, skillsSubimageResult, _ = subimageScaledSearch(equipmentImage, imageScale, SKILLS_TEMPLATE_IMAGE, skillsMaskImage)
    
    # get our skill count, depends on the slot count.
    skillCount = len(SKILL_SLOT_MASK_IMAGES)
    
    # initialize our return array. 0 represents the skill not being unlocked
    studentSkills = np.zeros(skillCount, int)
    
    # go through all of our skill slots
    for skill in range(skillCount):
        # get the mask for the respective slot
        skillSlotMaskImage = SKILL_SLOT_MASK_IMAGES[skill]
        
        # get the subimage of the skill slot
        skillSlotSubimage = cropImageWithMask(skillsSubimage, skillSlotMaskImage)
        cv2.imshow("skillSlotSubimage", skillSlotSubimage)
        # variables to keep track of our best skill level match
        bestSkillLevelResult = float("inf")
        bestSkillLevel = -1
        
        # depending on the slot, the max skill level is different. slot 1 only goes up to 5. slot 2-4 go up to 10
        skillLevelCount = SKILL_LEVEL_COUNT[skill]
        
        # go through all our skill level templates
        for skillLevel in range(skillLevelCount):
            # get our respective skill level template and mask
            skillLevelTemplate = SKILL_LEVEL_TEMPLATE_IMAGES[skillLevel]
            skillLevelMask = SKILL_LEVEL_MASK_IMAGES[skillLevel]

            cv2.imshow("skillLevelTemplate", skillLevelTemplate)
            cv2.imshow("skillLevelMask", skillLevelMask)
            
            # TM for the skill level template and get the result
            _, _, _, skillLevelResult, _ = subimageScaledSearch(skillSlotSubimage, imageScale, skillLevelTemplate, skillLevelMask)
            
            # check if our match result is better than our current best. if so update our variables
            if skillLevelResult < bestSkillLevelResult:
                bestSkillLevelResult = skillLevelResult
                bestSkillLevel = skill
        
        # we store "MAX" at skill level 0 because it's used in all skill slots, and there's no "0" for any
        if bestSkillLevel == 0:
            studentSkills[skill] = skillLevelCount
        else:
            studentSkills[skill] = bestSkillLevel
    
    # return our list of student's skill levels
    return studentSkills



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
            ueLevel = getLevels(ueSubimage, imageScale, UE_LEVEL_MASK_IMAGE, UE_LEVEL_TEMPLATE_IMAGES, UE_LEVEL_MASK_IMAGES, UE_LEVEL_MATCH_THRESHOLD, UE_LEVEL_OVERLAP_THRESHOLD)
    
    
    # return our ue's star and level
    return ueStar, ueLevel


# given the gear image, returns the level of the tier of the gear
def getGearTier(gearImage, imageScale):
    # given the specific gear image (gear1, gear2, or gear3), we get the subimage of where the tier should be
    gearTierSubimage = cropImageWithMask(gearImage, GEAR_TIER_MASK_IMAGE)
    
    # variables to keep track of which tier template matches best with our image
    bestTierResult = float("inf")
    bestTier = 0
    
    # calculate the amount of available tiers
    tierCount = len(TIER_LEVEL_TEMPLATE_IMAGES)
    
    # go through all the tier templates and see which one matches best 
    for tier in range(tierCount):
        # get the current tier template image
        tierTemplateImage = TIER_LEVEL_TEMPLATE_IMAGES[tier]
        
        # template match the current tier template in our subimage
        _, _, _, tierSubimageResult, _ = subimageScaledSearch(gearTierSubimage, imageScale, tierTemplateImage, TIER_LEVEL_MASK_IMAGE)
        
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
    studentGears = np.zeros(gearCount, int)
    
    # go through all of our individual gears
    for gear in range(gearCount):
        # get the current gear's level req
        gearSlotLevelRequirement = GEAR_SLOT_LEVEL_REQUIREMENTS[gear]
        
        # if student level isn't high enough, we break
        if studentLevel < gearSlotLevelRequirement:
            break
        
        # get the respective gear mask
        gearSlotMaskImage = GEAR_SLOT_MASK_IMAGES[gear]
        
        # get the gear subimage
        gearSubimage = cropImageWithMask(gearsSubimage, gearSlotMaskImage)
        
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





def checkInfo(studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers):
    if studentName != NAME_CHEAT_SHEET[COUNTER]:
        print(studentName, "|", NAME_CHEAT_SHEET[COUNTER])
        if studentBond != BOND_CHEAT_SHEET[COUNTER]:
            print(studentBond, "|", BOND_CHEAT_SHEET[COUNTER])
            if studentLevel != LEVEL_CHEAT_SHEET[COUNTER]:
                print(studentLevel, "|", LEVEL_CHEAT_SHEET[COUNTER])
                if studentStar != STAR_CHEAT_SHEET[COUNTER]:
                    print(studentStar, "|", STAR_CHEAT_SHEET[COUNTER])
                    if studentSkills != SKILL_LEVEL_CHEAT_SHEET[COUNTER]:
                        print(studentSkills, "|", SKILL_LEVEL_CHEAT_SHEET[COUNTER])
                        if ueStar != UE_STAR_CHEAT_SHEET[COUNTER]:
                            print(ueStar, "|", UE_STAR_CHEAT_SHEET[COUNTER])
                            if ueLevel != UE_LEVEL_CHEAT_SHEAT[COUNTER]:
                                print(ueLevel, "|", UE_LEVEL_CHEAT_SHEAT[COUNTER])
                                if gearTiers != GEAR_TIERS_CHEAT_SHEET_ARRAY[COUNTER]:
                                    print(gearTiers, "|", GEAR_TIERS_CHEAT_SHEET_ARRAY[COUNTER])
                            
                        


##directory = "student example"
##scale = 1.0
##for fileName in os.listdir(directory):
##    f = os.path.join(directory, fileName)
##    
##    sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)
##    
####    scale = SCALE_CHEAT_SHEET[COUNTER]
####    studentStar = STAR_CHEAT_SHEET[COUNTER]
####    studentLevel = LEVEL_CHEAT_SHEET[COUNTER]
##        
##    print(f, scale)
##    
##    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
##    cv2.imshow("equipmentImage", equipmentImage)
##    
##    studentName, studentBond, studentLevel, studentStar = getStudentStats(sourceImage, scale)
##    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
##    ueStar, ueLevel = getStudentUE(equipmentImage, scale, studentStar)
##    gearTiers = getStudentGears(equipmentImage,scale,studentLevel)
##    
##    checkInfo(studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers)
##    
##    COUNTER += 1

def main(sourceImage):
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
    
    studentName, studentBond, studentLevel, studentStar = getStudentStats(sourceImage, scale)
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
    ueStar, ueLevel = getStudentUE(equipmentImage, scale, studentStar)
    gearTiers = getStudentGears(equipmentImage,scale,studentLevel)
    
    checkInfo(studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers)

    
    return 1
