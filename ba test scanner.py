import os
import Constants
import cv2
import time
import random
import numpy as np
from pytesseract import pytesseract
from Constants import *
from Image import *


MAX_HIT = 0
MIN_MISS = float("inf")

COUNTER = 0

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


# docstrings example
#    """Add up two integer numbers.
#
#    This function simply wraps the ``+`` operator, and does not
#    do anything interesting, except for illustrating what
#    the docstring of a very simple function looks like.
#
#    Parameters
#    ----------
#    num1 : int
#        First number to add.
#    num2 : int
#        Second number to add.
#
#    Returns
#    -------
#    int
#        The sum of ``num1`` and ``num2``.
#
#    See Also
#    --------
#    subtract : Subtract one integer from another.
#
#    Examples
#    --------
#    >>> add(2, 2)
#    4
#    >>> add(25, 0)
#    25
#    >>> add(10, -10)
#    0
#    """

# THINGS LEARNED AND USED
# template matching
#    multi scale matching
#    multi object matching
#    scaled search
#    methods
# non-maximum suppression
# masks
# api for database stuff
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
# docstrings 



####### THINGS TO DO AND NOTES ########
# program can currently read a folder of screenshots, and even that of different resolution and sizes
# redo thresholds and crap, especially for skill level
# video scanner
#   bluestacks screen recoridng records at 1280x720, so if using that please make your client the same resolution
#       the reason is because if you're doing a different aspect ratio, the recording will have squeezed/stretched items
# website.www
# item/inventory scanner
#   elephs
#       
# MIGHT NEED TO ADD AN ERROR CATCHER FOR WHEN A SS IS GIVEN AND THERE'S NOTHING MATCHING/DOESN'T MEET THRESHOLD




# from an array of results, locations, and sizes, we filter the results that
# overlap with our best matches, guaranteeing unique results
def nms(matchResults, matchLocations, matchWidth, matchHeight, overlapThreshold):
    """From arrays of results and respective info, find the best, unique results.

    We take our best matches and then compare the area they cover with the other results. If any overlap passes
    our threshold, we remove them. Then we move onto the next best result and keep going til we get to the end
    of our array or there are no more results.
    Our threshold was determined through a bunch of testing.

    Returns
    nmsResults:   array of the best, final, filtered results
    nmsLocations: locations of those results. each value should be a pair that represents
                  the location on the image 
    nmsCount:     amount of filtered results
    nmsOrder:     i believe this variable was used for overlap testing, don't see any current use in the program
    """
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
    """Filter our results and run them through nms to find the best, unique matches, then return them

    First do an easy filter where we remove the values that don't pass an initial threshold.
    Second, create an array of (x,y) values representing the filtered results' locations in the image.
    Finally we run these results through nms, which should give us the best, unique matches

    Returns
    nmsResults:   array of the best, final, filtered results
    nmsLocations: locations of those results. each value should be a pair that represents
                  the location on the image 
    nmsCount:     amount of filtered results
    nmsOrder:     i believe this variable was used for overlap testing, don't see any current use in the program
    """
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
    
    nmsCount = len(nmsLocations)
    
    for index in range(nmsCount):
        colorr = np.array([0, 0, 255]) * (nmsCount - index) / nmsCount
        location = nmsLocations[index]
        cv2.rectangle(sourceImage, (location), (x2[index], y2[index]), colorr, 1)

    cv2.imshow(str(time.time()), scaleImage(sourceImage, 5.0))

    return 0


# template match given a scale. we scale the template and optional mask, then TM. keep the source as is.
# used for when you know what the scale of the template will be in your source image.
def subimageScaledSearch(colorSourceImage, imageScale, colorTemplateImage, maskImage = None):
    """Scaling a template image with an optional mask, search through the source image to find the best
    match to the template.

    With our given scale, scale the template image and search through the image to find the best match.

    Returns
    subimage:     subimage of our matched template
    scaledWidth:  width of our subimage
    scaledHeight: height of our subimage
    minVal:       value from the calculation of the best result
    matchResults:  array of all the values of our template match search
    """
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
        matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
    else:
        scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
        matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
    
    # extract our values from the template match result
    minVal, _, minLoc, _ = cv2.minMaxLoc(matchResults)
    
    # compute the area of our subimage in our source image
    x1 = minLoc[0]
    x2 = x1 + scaledWidth
    y1 = minLoc[1]
    y2 = y1 + scaledHeight
    
    # get our subimage
    subimage = colorSourceImage[y1:y2, x1:x2]

    # return the subimage along with some other values
    return subimage, scaledWidth, scaledHeight, minVal, matchResults


# resize our template image at different scales to try to find the best match in our source image
# return the results of said match, and the location of best match
def subimageMultiScaleSearch (colorSourceImage, colorTemplateImage, maskImage = None):
    """With a template image and optional mask, find the best scale of our template match in the source image
    by continuously matching while the scale.

    Template match at one scale. Record the result. Now do again with incrementing and decrementing scales
    until you have gone through all possibilities. Record and return our best match

    Returns
    subimage:         subimage of our best matched template
    bestWidth:        width of our subimage
    bestHeight:       height of our subimage
    bestScale:        scale that we found to result in the closest template match
    bestMatchResults: array of all the values of our template match search when using our best scale
    """
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
    bestMatchResults = []
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
            matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
        else:
            scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
            matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
##            matchResult, method = runAllTemplateMatchingMethods(graySourceImage, scaledGrayTemplateImage, scaledMaskImage)

        # store our values into corresponding variables
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(matchResults)

        # check to see if our current match is better than our best match
        if minVal < bestMatch:
            # if so, update our tracking variables and reset our bad counter
            bestMatch = minVal
            bestMatchLocation = minLoc
            bestWidth = scaledWidth
            bestHeight = scaledHeight
            bestScale = scale
            bestMatchResults = matchResults
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
            matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD)
        else:
            scaledMaskImage = cv2.resize(maskImage, (scaledWidth, scaledHeight))
            matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
        
        minVal, _, minLoc, _ = cv2.minMaxLoc(matchResults)

        # check to see if our current match is better than our best match
        if minVal < bestMatch:
            # if so, update our tracking variables and reset our bad counter
            bestMatch = minVal
            bestMatchLocation = minLoc
            bestWidth = scaledWidth
            bestHeight = scaledHeight
            bestScale = scale
            bestMatchResults = matchResults
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
    
    return subimage, bestWidth, bestHeight, bestScale, bestMatchResults


# get the name of the student given the stats image
def getStudentName(statsImage, imageScale):
    """Given the stats subimage and the scale, process and return the student's name."""
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
##    cv2.imshow("sourceLevelsImage", sourceLevelsImage)
    
    # lists to keep track of our matches
    levelsWidths = []            # widths of the TM'ed level subimages
    levelsHeights = []           # heights of the TM'ed level subimages
    levelsNMSResults = []   # nms-filtered match results that passed our threshold
    levelsNMSLocations = [] # locations of said matches
    levelsNMSMatches = []   # the level template of the match (0-9)
    
    # go through range 0-9 and tm for the respective level template
    levelCount = len(levelTemplateImages)
##    print("levelCount", levelCount)
    for level in range(levelCount):
        # get the current level template and mask
        levelTemplateImage = levelTemplateImages[level]
        levelMaskImage = levelMaskImages[level]

##        cv2.imshow("levelTemplateImage", levelTemplateImage)
##        cv2.imshow("levelMaskImage", levelMaskImage)
        
        # TM for it in our levelsSubimage
        levelSubimage, levelSubimageWidth, levelSubimageHeight, levelSubimageResult, levelSubimageResults = subimageScaledSearch(sourceLevelsImage, imageScale, levelTemplateImage, levelMaskImage)
##        print(levelSubimageResult)
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
            nmsResults, nmsLocations, nmsCount, _ = filterResultsAndLocations(levelSubimageResults, levelSubimageWidth, levelSubimageHeight, matchThreshold, overlapThreshold)
            
            ## record our results
            # the reason why we extend widths, heights, and matches multiple times is because
            # results and locations may have multiple values. we want all these arrays to
            # have the same length for when we use them later in nms.
            levelsWidths.extend([levelSubimageWidth] * nmsCount)
            levelsHeights.extend([levelSubimageHeight] * nmsCount)
            levelsNMSResults.extend(nmsResults)
            levelsNMSLocations.extend(nmsLocations)
            levelsNMSMatches.extend([level] * nmsCount)
    
    # convert our lists to np arrays for coding purposes
    levelsWidths = np.array(levelsWidths)
    levelsHeights = np.array(levelsHeights)
    levelsNMSResults = np.array(levelsNMSResults)
    levelsNMSLocations = np.array(levelsNMSLocations)
    levelsNMSMatches = np.array(levelsNMSMatches)
    
    # further filter our results. do so because sometimes numbers like "1" will TM to "4"s, and "5"s and "6"s, or "3"s and "8"s.
    # so with our array of all our matches thrown together, we filter it to make sure that we only have the best matches + no overlap.
    # a 3 should have a better match to 3 than 8 would to 3, so the 3 has priority. then we filter matches that have significatn overlap with the 3 (the 8)
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
    
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
##    print(nms2Results)
    
    
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
##    cv2.imshow("stats sub", statsSubimage)
    
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
    skillsCount = len(SKILLS_SLOT_STAR_REQUIREMENTS)
    
    # initialize our return array. 0 represents the skill not being unlocked
    studentSkills = np.zeros(skillsCount, int)
    
    # go through all of our skill slots
    for skillsSlot in range(skillsCount):
        # get the star requirement for the respective skill slot
        skillsSlotLevelRequirement = SKILLS_SLOT_STAR_REQUIREMENTS[skillsSlot]
        
        # if student star isn't high enough, we break
        if studentStar < skillsSlotLevelRequirement:
            break
        
        #
        skillsSlotLevelMax = SKILLS_SLOT_LEVEL_MAX[skillsSlot]
##        used this to try to figure out why my stuff broke with a non-ipad image. was cause my threshold was way too low when looking at smaller images
##        cv2.imshow("skillsubimage" + str(skillsSlot), skillsSubimage)
##        print(imageScale)
##        cv2.imshow("slot mask" + str(skillsSlot), SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot])
##        for asdf in range(skillsSlotLevelMax):
##            cv2.imshow(str(asdf) + "t" + str(skillsSlot), SKILL_LEVEL_TEMPLATE_IMAGES[asdf])
##            cv2.imshow(str(asdf) + "m" + str(skillsSlot), SKILL_LEVEL_MASK_IMAGES[asdf])
        
        # else, we get the level for the respective skill slot
        skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.05, 0.665)

        if skillsLevel == 0:
            studentSkills[skillsSlot] = skillsSlotLevelMax
        else:
            studentSkills[skillsSlot] = skillsLevel
        
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
##    cv2.imshow(str(time.time()) + "geartiersubimage", gearTierSubimage)
    
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
        tierSubimage, tierSubimageWidth, tierSubimageHeight, tierSubimageResult, tierSubimageResults = subimageScaledSearch(gearTierSubimage, imageScale, tierTemplateImage, TIER_LEVEL_MASK_IMAGE)
##        print("tier" + str(tier) + ":" + str(tierSubimageResult))
        # check to see if our current match is better than our best match
        # update our variables if so
        if tierSubimageResult < bestTierResult:
##            cv2.imshow(str(time.time()) + "tierSubimage" + str(tier), tierSubimage)
            bestTierResult = tierSubimageResult
            bestTier = tier + 1 # +1 because counter starts at 0
    
    # return our best-matched tier
    return bestTier


# given the equipment subimage, return an array of the student's gears tiers
def getStudentGears(equipmentImage, imageScale, studentLevel):
    # given our equipment image, TM for the gears area
    gearsSubimage, _, _, gearsSubimageResult, _ = subimageScaledSearch(equipmentImage, imageScale, GEARS_TEMPLATE_IMAGE, GEARS_MASK_IMAGE)    
    
    # get the gear slot count
    gearCount = len(GEARS_SLOT_LEVEL_REQUIREMENTS)

    # array to keep track of our student's gear's tiers.
    # we premake it because even if student doesnt meet the level req,
    # that means the tier is 0, or in other words nothing is equipped
    studentGears = np.zeros(gearCount, int)
    
    # go through all of our individual gears
    for gear in range(gearCount):
        # get the current gear's level req
        gearsSlotLevelRequirement = GEARS_SLOT_LEVEL_REQUIREMENTS[gear]
        
        # if student level isn't high enough, we break
        if studentLevel < gearsSlotLevelRequirement:
            break
        
        # get the respective gear mask
        gearsSlotMaskImage = GEARS_SLOT_MASK_IMAGES[gear]
        
        # get the gear subimage
        gearSubimage = cropImageWithMask(gearsSubimage, gearsSlotMaskImage)
##        cv2.imshow("gearSubimage" + str(gear), gearSubimage)
        
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
                            
                        

##
##directory = "student example"
##scale = 1.0
##for fileName in os.listdir(directory):
##    f = os.path.join(directory, fileName)
##    
##    sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)
##    
##    scale = SCALE_CHEAT_SHEET[COUNTER]
##    studentStar = STAR_CHEAT_SHEET[COUNTER]
##    studentLevel = LEVEL_CHEAT_SHEET[COUNTER]
##        
##    print(f, scale)
##
##    print( EQUIPMENT_TEMPLATE_IMAGE)
##    cv2.imshow("EQUIPMENT_TEMPLATE_IMAG", EQUIPMENT_TEMPLATE_IMAGE)
##    cv2.imshow("EQUIPMENT_MASK_IMAGE", EQUIPMENT_MASK_IMAGE)
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

def getStudentInfoFromImage(sourceImage):
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)

##    start = time.time()
    studentName, studentBond, studentLevel, studentStar = getStudentStats(sourceImage, scale)
##    print(studentName, studentBond, studentLevel, studentStar)
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
    ueStar, ueLevel = getStudentUE(equipmentImage, scale, studentStar)
    gearTiers = getStudentGears(equipmentImage, scale, studentLevel)
##    end = time.time()
    
##    print("levels time:" + str(end-start))
    
##    checkInfo(studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers)
    
    
    student = Student(studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers)
    
    
    return studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers



def main():
    os.chdir("..")
    directory = "student example"
    for fileName in os.listdir(directory):
        f = os.path.join(directory, fileName)
        
        print(f)
        sourceImage = cv2.imread(f, cv2.IMREAD_COLOR)
        
        startTime = time.time()
        studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = getStudentInfoFromImage(sourceImage)
        endTime = time.time()
        print(studentName, "bond", studentBond, ", level", studentLevel, studentStar,"star", studentSkills, ueStar, ueLevel, gearTiers)
        
        print("totaltime:" + str(endTime-startTime))



##### video scanner #####
### USER RECORDING INSTRUCTIONS ###
# if using bluestacks, there is a built in screen recorder that is very easy to use
# please use any of the default landscape resolutions, or any resolution that is 16:9 aspect ratio
# user will start with the student info open of any student before starting recording.
# start the recording
# click the right or left arrow on the screen in-game continuously, going through the students
# should not have to worry about pictures like splash art, gear image, and ue image loading.
#   all the info needed is loaded immediately, again do not worry. try to speedrun if you want.
#   in fact, i think it'll make it faster for me if you click through quickly
# do this until you eventually loop back to the initial character you had on screen before starting the recording
# stop the recording





# load a videofile given its name and go through it frame-by-frame. we grab the students' info from the frame
# returns a list of students' info
def suckFrames(videoFileName : str):
    """yes"""
    # basically load the video into this variable, acts similar to an array.
    videoCapture = cv2.VideoCapture(videoFileName)
    
    # check if the video was opened properly
    isOpened = videoCapture.isOpened()
    
    # continue if video was opened properly
    if isOpened == True:
        # get the first frame
        ret, frame = videoCapture.read()
        
        # create a set to keep track of the student names. set instead of list because order of names doesn't matter and
        # it's faster to check if an element exists in a set than in a list
        studentNameSet = set()
        
        #
        
        
        
        ## 
        # 
        while ret == True:
##            # look through the current frame to try to get student info
##            studentName = grabStudentName(frame) # this should return "" if there was no name
##            
##            ## if this frame's student's name doesn't exist in our set, nab the info and add them to the list
##            # check if the current student's name doesn't exist in our set
##            if studentName not in studentNameSet:
##                # grab the student info
##                studentInfo = grabStudentInfo(frame)
##                
##                # store our student info
##                studentSet.add(studentInfo)
##                
##                # get the next frame of the video
##                ret, frame = videoCapture.read()
            
            
        
        
        ##### should i make a thing where it's like it makes a mask based off of the first student? think i'll only apply it to the video. the reason why
        ##### i don't wanna add it to ss's is that how would i know? if i judge off scale, then what if their first ss is of their window, then next is
        ##### the window on their desktop. it should still have the same scale given that the only thing that changed was the size of the whole image.
        ##### if i do scale + resolution, then it could be the same scenario but they moved their window from left side to right side of screen.
        #### to find locations of specific items, you're basically just adding all the foujnd locations.
        ####    ex. eq wind is found at 200, 100. skill window in eq is found at 10, 100. skillslot1 is found at 5,50. then the location of skillslot1 relative
        ####    the source image would be (x1 + x2 + x3), (y1 + y2 + y3). then we just make a mask based on that. we'll also have the sizes of each one too so we wouldn't need
        ####    to keep looking 
        ## !get the first frame! DO NOT DO THIS ANYMORE. DIDN'T THINK ABOUT PHONE SCREEN RECORDER LIKE IPHONE WHERE THEY GO THROUGH THEIR MENU FIRST 
        ## !grab all of the info of the first student!
        ## !remember the name (and the location of the name?) and record this frame!
        ## go frame-by-frame
        ##  check the name of each frame
        ##  if new name
        ##      stop once you find a new
        
            
    
    # if nothing then something went wrong
    else:
        print("error opening")
    
##    # if successfully opened, go through the frames one-by-one and grab the student info in the respective
##    # student's last shown frame. this is to make sure that all their info is loaded in, as opposed to getting
##    # their info the first frame they're shown as the game sometimes has a delay before all info is loaded onscreen
##    while videoCapture.isOpened():
##        # grab the frame, also has a return value to see if anything was returned
##        ret, frame = videoCapture.read()
##        
##        # check if a frame was returned
##        if ret == True:
##            # if so,
##            
##        else:
##            break
##    videoCapture.release()


