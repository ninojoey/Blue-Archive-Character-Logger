import cv2
import random
import numpy as np
from pytesseract import pytesseract


SITE_VERSION = "1.3.5"
SCALE_INCREMENT = 0.001
STAT_WINDOW_MATCH_MINIMUM = 0.6
SKILL_WINDOW_MATCH_MINIMUM = 0.7
MAX_TEMPLATE_MATCH_MINIMUM = 0.9
TEMPLATE_MATCH_METHOD = cv2.TM_CCORR_NORMED




# indices
# [2:] = starting with index 2
# [:2] up to index 2
# [1:2] items from index 1 to 2
# [::2] = every 2 items, in other words skip 1
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


# logic for pulling from json
# take in import txt. check export and site version. check if it lines up with our
# copy of the database (variables inside database will be used to check)
# if lines up, continue with our database. if not, we update our database by
# running the json grabber thingie and write to the database.






def createBlankMask(width, height):
    # create a 2d array of dimensiions (height x width)
    # where each element is an array of three 1's ([1, 1, 1])
    # which are of type uint8 (0-255). then we multiply the
    # entire 2d array by 255, making it a 2d array of 255's
    blankMask = np.ones([height, width, 4], dtype = np.uint8) * 255

    return blankMask


def createTransparencyMask(img_rgb):
    transparencyMask = img_rgb.copy()

    for row in range(len(transparencyMask)):
        for col in range(len(transparencyMask[0])):
            if transparencyMask[row][col][3] < 255:
                transparencyMask[row][col] = [0, 0, 0, 255]
            else:
                transparencyMask[row][col] = [255, 255, 255, 255]

    return transparencyMask

# look through our image to find the best match of our template
# return the results of said match, and the location of best match
def subimageSearch (img_rgb, template_rgb, mask_img = []):
    # make a grayscale copy of the main image, and store its width and height
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img_gray.shape

    # make a grayscale copy of the template image, and store its width and height
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
    template_height, template_width  = template_gray.shape

    if len(mask_img) == 0:
        mask_img = createTransparencyMask(template_rgb)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)


    # find the best match of our template in our og iamge. we match and check its match value.
    # we scale the template up until we match one of the original's dimesions.
    
    # variable to scale the template by
    scale = 1.0

    # variables that keep track of values of our best match
    best_match = 0
    best_location = (0,0)
    best_width = template_width
    best_height = template_height
    best_scale = 1.0
    best_result = 0

    # values of our newly scaled template/image
    resized_width  = template_width
    resized_height = template_height


    # scale up our template and match to look for the best match until
    # we hit one of original's dimensions
    while resized_width <= img_width and resized_height <= img_height:
        # resize our template after we've scaled it
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))
        resized_mask = cv2.resize (mask_img, (resized_width, resized_height))
        
        # find the location of our best match to the template in our resized image.
        # matchTemplate returns a 2d array of decimals. the decimal represents the match value
        # of the tempalte at that respective location on the image.
        result = cv2.matchTemplate(img_gray, resized_gray, TEMPLATE_MATCH_METHOD, mask = resized_mask)

        # store our values into corresponding variables
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # check to see if our current match is better than our best match    
        if max_val > best_match:
            # if so, update our tracking variables
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
            best_scale = scale
            best_result = result

        # increment our scale and adjust our width and height
        scale = scale + SCALE_INCREMENT
        resized_width  = int ( template_width * scale )
        resized_height = int ( template_height * scale )


    # if best_match == 0, that means our template was bigger than our original, ignoring
    # through the first loop entirely without doing anything. so now we make our template
    # image small enough for the original to be searched through.
    # the reason we do this after our first loop is because the first loop scales up the 
    # template image. and if our template is already too big, we can just skip that part 
    # entirely. but in all scenarios, we still wanna scale down in case we can find a better match.

    # if we skipped the first loop entirely, that means our template is bigger than our source
    # so we scale the template down to match the source's
    if best_match == 0:
        # find which side to scale off of. will be based on whichever difference is bigger
        width_difference = template_width - img_width
        height_difference = template_height - img_height

        # we scale our images, and start our scale here
        if width_difference >= height_difference:
            scale = img_width / template_width
        else:
            scale = img_height / template_height


    # just reset our variables if we didn't need to scale up our original image
    else:
        scale = 1.0

    resized_width  = int ( template_width * scale )
    resized_height = int ( template_height * scale )

    # now we look through the image by scaling down the template
    # since we're scaling down, we don't have to worry about the og's dimension limits.
    # so instead we set the limit of how many bad matches we get in succession and
    # make sure our scale doesnt go below 0
    # same code as our first loop except we increment our scale instead of decrementing

    xBoundary = img_width * 0.1
    yBoundary = img_height * 0.1
    
    
    while resized_width > xBoundary and resized_height > yBoundary:
        
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))
        resized_mask = cv2.resize(mask_img, (resized_width, resized_height))

        result = cv2.matchTemplate(img_gray, resized_gray, TEMPLATE_MATCH_METHOD, mask = resized_mask)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_match:
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
            best_scale = scale
            best_result = result


        scale = scale - SCALE_INCREMENT
        resized_width  = int ( template_width * scale )
        resized_height = int ( template_height * scale )
    


    # at this point, we should now have the values of our best match. now return it cropped out

    
    # get a cropped image of our matched area from the original image
    subimage_x1 = best_location[0]
    subimage_x2 = subimage_x1+ best_width
    subimage_y1 = best_location[1]
    subimage_y2 = subimage_y1 + best_height
    subimage = img_rgb[subimage_y1:subimage_y2, subimage_x1:subimage_x2]

    print(best_match)
    
    return subimage, best_width, best_height, best_scale, best_result




# mean squared error formula/function
def mse(img1_rgb, img2_rgb, mask = []):
    if len(mask) == 0:
        mask = createTransparencyMask(img2_rgb)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)
    
    # store our image's width and height. they should be the same
    img_width = img1_gray.shape[1]
    img_height = img1_gray.shape[0]

    img1_masked = cv2.bitwise_and(img1_gray, img1_gray, mask = mask)

    # subtract our image. will return a 2d array of 0's and 1's
    # 0 should represent if the pixels are the same. 1 if not
    diff_img = cv2.subtract(img1_masked, img2_gray)

    # sum up all the diff squared
    diff = np.sum(diff_img**2)

    # calculate the mse
    error = diff/(float(img_height*img_width))
    
    return error, diff_img




def returnErrAndDiff(template_img, subimage):
    # make sure both images are the same dimensions
    template_resized = cv2.resize(template_img, (subimage.shape[1], subimage.shape[0]))
    template_resized_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)

    subimage_gray = cv2.cvtColor(subimage, cv2.COLOR_BGR2GRAY)
    
    error, diff = mse(template_resized_gray, subimage_gray)
    print("Image matching Error between the two images:", error)

    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(diff)
    print (text[:-1])
    
    return diff



def nms(results, boxesCoordinates, boxWidth, boxHeight):
    resultsCopy = np.copy(results)

    # unpack our arrays
    boxes_x1 = boxesCoordinates[1]
    boxes_x2 = boxes_x1 + boxWidth
    boxes_y1 = boxesCoordinates[0]
    boxes_y2 = boxes_y1 + boxHeight
    
    # determine which order to run through the results/coordinates.
    # we do it based on the results, first being the highest value.
    # but since argsort sorts from low to high, we reverse it with [::-1]
    # we create a dedicated array instead of just sorting the results array itself
    # since we're working with other arrays (the boxes array which has x and y coordinates)
    indexOrder = np.argsort(results)[::-1]
    
    # get the area of the box. they should all be the same area
    boxArea = boxWidth * boxHeight

    # create an array to store our overlap percentages
    overlap = []

    # array to store the coordinates of our best matches, filtering out overlappers
    filteredCoordinates = []

    # go through all our boxes starting with highest match to lowest match.
    # grab the coordinates of that box and store it into our filtered list.
    # go through the rest of the coordinates to see if any of them have overlap.
    # if any have overlap over the threshhold, we delete those values from our list
    while len(indexOrder) > 0:
        # get the index to work with. should be our current highest match result
        index = indexOrder[0]

        # get the coordinates of that match and add it to our filtered list
        best_x1 = boxes_x1[index]
        best_x2 = boxes_x2[index]
        best_y1 = boxes_y1[index]
        best_y2 = boxes_y2[index]
        filteredCoordinates.append((best_x1, best_y1))

        # determine the overlap of the other boxes with the current match.
        # to find the overlapping area, the x and y values should be furthest away
        # from the edges of the original.
        overlap_x1 = np.maximum(best_x1, boxes_x1)
        overlap_x2 = np.minimum(best_x2, boxes_x2)
        overlap_y1 = np.maximum(best_y1, boxes_y1)
        overlap_y2 = np.minimum(best_y2, boxes_y2)

        # we do a max of 0, because if you have a negative edges, that means
        # there isn't any overlap, and if you have 2 negative edges, the area will
        # still be positive, resulting in a false positive.
        overlapWidth = np.maximum(0, overlap_x2 - overlap_x1)
        overlapHeight = np.maximum(0, overlap_y2 - overlap_y1)
        overlapArea = overlapWidth * overlapHeight

        # determine the percentage of overlap with our original box area
        overlap = overlapArea / boxArea

        # delete the boxes where the overlap is over a certain threshold.
        # in this case, we delete the entries of the indicies so we don't
        # go over them later. also delete the one we just worked on cause it's done
        indexDelete = np.where(overlap>0.9)[0]
        indexDelete = np.append(indexDelete, 0)
        indexOrder = np.setdiff1d(indexOrder, indexDelete)

    # return the coordinates of the filtered boxes
    return filteredCoordinates


def main():
    # load our original image + our template of the skill window
    img_rgb = cv2.imread('hibiki beeg.png')
    statWindowTemplate_rgb = cv2.imread('stat window blank.png')

    # get the cropped stat window of our matched template
    statWindowSubimage_rgb,_,_,_ = subimageRGBSearch(img_rgb, statWindowTemplate_rgb)
    cv2.imshow("statWindowSubimage_rgb", statWindowSubimage_rgb)

    # load in the skill window template image and get the cropped subimage 
    skillWindowTemplate_rgb = cv2.imread('skill window blank.png')
    skillWindowSubimage_rgb,_,_,_ = subimageRGBSearch(statWindowSubimage_rgb, skillWindowTemplate_rgb)
    cv2.imshow("skillWindowSubimage_rgb", skillWindowSubimage_rgb)

    # load in a copy of skill window and show the difference
    # certain areas of skill window are painted over so it's possible to read the specific text i want
    negativeSkillTemplate_gray = cv2.imread('skill window focus.png')
    #skillDifference = returnErrAndDiff(negativeSkillTemplate_gray, skillWindowSubimage_rgb)
    #cv2.imshow("skill diff", skillDifference)

    # since one of the objects in the skill window is an image and not text, i'll load in a copy of said object
    maxSkillTemplate_rgb = cv2.imread('max template.png')
    cv2.imshow("max maxSkillTemplate_rgb", maxSkillTemplate_rgb)
    #max_subimage = findSubimageInImage(subimage, max_template)
    #printAndShowErrDiff(max_template, max_subimage)

    # get the result of our template search
    maxSkillSubimage_rgb, boxWidth, boxHeight, match_results = subimageRGBSearch(skillWindowSubimage_rgb, maxSkillTemplate_rgb)
    cv2.imshow("maxSkillSubimage_rgb", maxSkillSubimage_rgb)

    # match_result is a 2d array where each value is the tempalte's match percentage to
    # the original image at that given coordinate on the image. so the result at [0][0]
    # corresponds to the template's match percentage at (0,0) on the image.
    # sift through our result and get the ones that are above our threshold
    filteredCoordinates = np.where(match_results >= 0.80)
    filteredMatchesResults = np.extract(match_results >= 0.80, match_results)

    # run our coordinates and results through NMS. now we should have coordinates of the 
    # best results with little-to-no overlapping areas
    nmsCoordinates = nms(filteredMatchesResults, filteredCoordinates, boxWidth, boxHeight)


    # now with nms done, we have to determine where those max's belong to and where levels would belong to



    #gearWindowTemplate_rgb = cv2.imread("T6_Hairpin.png", cv2.IMREAD_UNCHANGED)
    #gearWindowTemplate_gray = cv2.cvtColor(gearWindowTemplate_rgb, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gearWindowTemplate_gray", gearWindowTemplate_gray)
    #gearWindowSubimage_rgb,_,_,_ = subimageSearch(statWindowSubimage_rgb, gearWindowTemplate_rgb)
    #cv2.imshow("gearWindowSubimage_rgb", gearWindowSubimage_rgb)

    


def recolor(img):
    for row in range(len(img)):
        for pixel in range(img.shape[1]):
            if not (img[row][pixel][0] == 100 and img[row][pixel][1] == 70 and img[row][pixel][2] == 45):
                img[row][pixel][0] = 255
                img[row][pixel][1] = 255
                img[row][pixel][2] = 255

    return img



def combineTransparentImages(img1, img2, xOffset = 0, yOffset = 0):
    img2Width = img2.shape[1]
    img2Height = img2.shape[0]
    
    x1 = xOffset
    x2 = x1 + img2Width
    y1 = yOffset 
    y2 = y1 + img2Height

    # get the alpha channel of img2 and turn it into a scalar. max value of it is 255
    img2Alpha = img2[:,:,3] / 255.0
    # calculate what img1's alpha should be. it is img2's complement 
    img1Alpha = 1.0 - img2Alpha

    # make a copy to work with
    combinedImage = img1.copy()

    # apply the scalar to 
    for BGRAIndex in range (0, 3):
        print(img2[:, :, BGRAIndex])
        print(img2Alpha * img2[:, :, BGRAIndex])
        combinedImage[y1:y2, x1:x2, BGRAIndex] = img2Alpha * img2[:, :, BGRAIndex] + img1Alpha * img1[y1:y2, x1:x2, BGRAIndex]

    return combinedImage




# import images
s_img = cv2.imread("T7_Shoes.png", -1)
card_img = cv2.imread("Card_Bg.png", -1)
l_imga = cv2.resize(card_img, (312, 256))

# my calculated, hard-coded offset for combining our iamges
xOffset = 4
yOffset = 6

# combine our images
l_img = combineTransparentImages(l_imga, s_img, xOffset, yOffset)
cv2.imshow("b4", l_img)

# read in our mask
mask_img = cv2.imread("gear mask.png", -1)
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
mask_img = cv2.resize(mask_img, (312, 256))


# read in our source image
akashoes = cv2.imread("akane shoes.png", -1)


# template match for our combbined image with our mask
shoesSub, best_width, best_height, best_scale, best_result = subimageSearch(akashoes, l_img, mask_img)

# update our image to match our subimage's size
l_img = cv2.resize(l_img, (best_width, best_height))
cv2.imshow("shoesSub", shoesSub)

shoesSub_gray = cv2.cvtColor(shoesSub, cv2.COLOR_BGR2GRAY)


gray = shoesSub_gray.copy()
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
cv2.imshow('threshpre', thresh)

startPoint = (0, 0)
newColor = (0, 0, 0, 255)
cv2.floodFill(thresh, None, startPoint, newColor)
cv2.imshow('thresh', thresh)



levelMask = cv2.imread("level mask.png", -1)
levelMask = cv2.cvtColor(levelMask, cv2.COLOR_BGR2GRAY)
levelMask = cv2.resize(levelMask, (best_width, best_height))



thresh = cv2.bitwise_and(thresh, levelMask)
cv2.imshow('thresh', thresh)


path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract
text = pytesseract.image_to_string(thresh)
print("text:", text)


