import cv2
import numpy as np
from pytesseract import pytesseract
from Constants import * 



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
