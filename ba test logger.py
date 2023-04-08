import cv2
import numpy as np
from pytesseract import pytesseract



SCALE_INCREMENT = 0.01
STAT_WINDOW_MATCH_MINIMUM = 0.6
SKILL_WINDOW_MATCH_MINIMUM = 0.7
MAX_TEMPLATE_MATCH_MINIMUM = 0.9


# things learned/used
# template matching
#    multi scale matching
#    multi object matching
# non-maximum suppression


# rethink how subimage serach should work. sometimes it takes too long. sometimes if you go too small template then it'll match cause it'll be the size of a pixel
# should we try to add hard barriers depending on which item we're working on? ughhhhhhhhhhhhhh


# what to do after nms
# figure out which skills are maxed and which are leveled
# maybe divide up the skill subwindow into segments. if max coordinate falls into x segment, it's for ex/passive/etc
# still have to get the name, level, exp, stars, weapon, gear, oh my god lol



# look through our image to find the best match of our template
# return the results of said match, and the location of best match
def subimageSearch (img_rgb, template_rgb):
    # make a grayscale copy of the main image, and store its width and height
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_width = img_gray.shape[1]
    img_height = img_gray.shape[0]

    # make a grayscale copy of the template image, and store its width and height
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
    template_width  = template_gray.shape[1]
    template_height = template_gray.shape[0]


    # find the best match of our template in our og iamge. we match and check its match value.
    # we scale the template up until we match one of the original's dimesions.
    
    # variable to scale the template by
    scale = 1.0

    # variables that keep track of values of our best match
    best_match = 0
    best_location = (0,0)
    best_width = template_width
    best_height = template_height
    best_result = 0

    # values of our newly scaled template/image
    resized_width  = template_width
    resized_height = template_height


    # scale up our template and match to look for the best match until
    # we hit one of original's dimensions
    while resized_width <= img_width and resized_height <= img_height:
        # resize our template after we've scaled it
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))
        
        # find the location of our best match to the template in our resized image.
        # matchTemplate returns a 2d array of decimals. the decimal represents the match value
        # of the tempalte at that respective location on the image.
        result = cv2.matchTemplate(img_gray, resized_gray, cv2.TM_CCOEFF_NORMED)

        # store our values into corresponding variables
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # check to see if our current match is better than our best match    
        if max_val > best_match:
            # if so, update our tracking variables
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
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

    # if we skipped the first loop entirely, that means we need to resize our template image
    if best_match == 0:
        # find which side to scale off of. will be based on whichever difference is bigger
        width_difference = template_width - img_width
        height_difference = template_height - img_height

        # we scale our images, and start our scale here
        if width_difference >= height_difference:
            scale = img_width / template_width
        else:
            scale = img_height / template_height

        # resize according to our scale
        resized_width  = int ( template_width * scale )
        resized_height = int ( template_height * scale )


    # just reset our variables if we didn't need to scale up our original image
    else:
        scale = 1.0
        resized_width  = template_width
        resized_height = template_height
    

    # now we look through the image by scaling down the template
    # since we're scaling down, we don't have to worry about the og's dimension limits.
    # so instead we set the limit of how many bad matches we get in succession and
    # make sure our scale doesnt go below 0
    # same code as our first loop except we increment our scale instead of decrementing

    xBoundary = img_width * 0.01
    yBoundary = img_height * 0.01
    
    while resized_width > xBoundary and resized_height > yBoundary:
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))

        result = cv2.matchTemplate(img_gray, resized_gray, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
 
        if max_val > best_match:
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
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
    print (best_match)
    return subimage, best_width, best_height, best_result




def subimageRGBSearch (img_rgb, template_rgb):
    # make a grayscale copy of the main image, and store its width and height
    img_gray = img_rgb
    img_width = img_gray.shape[1]
    img_height = img_gray.shape[0]

    # make a grayscale copy of the template image, and store its width and height
    template_gray = template_rgb
    template_width  = template_gray.shape[1]
    template_height = template_gray.shape[0]


    # find the best match of our template in our og iamge. we match and check its match value.
    # we scale the template up until we match one of the original's dimesions.
    
    # variable to scale the template by
    scale = 1.0

    # variables that keep track of values of our best match
    best_match = 0
    best_location = (0,0)
    best_width = template_width
    best_height = template_height
    best_result = 0

    # values of our newly scaled template/image
    resized_width  = template_width
    resized_height = template_height


    # scale up our template and match to look for the best match until
    # we hit one of original's dimensions
    while resized_width <= img_width and resized_height <= img_height:
        # resize our template after we've scaled it
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))
        
        # find the location of our best match to the template in our resized image.
        # matchTemplate returns a 2d array of decimals. the decimal represents the match value
        # of the tempalte at that respective location on the image.
        result = cv2.matchTemplate(img_gray, resized_gray, cv2.TM_CCORR_NORMED)

        # store our values into corresponding variables
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # check to see if our current match is better than our best match    
        if max_val > best_match:
            # if so, update our tracking variables
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
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

    # if we skipped the first loop entirely, that means we need to resize our template image
    if best_match == 0:
        # find which side to scale off of. will be based on whichever difference is bigger
        width_difference = template_width - img_width
        height_difference = template_height - img_height

        # we scale our images, and start our scale here
        if width_difference >= height_difference:
            scale = img_width / template_width
        else:
            scale = img_height / template_height

        # resize according to our scale
        resized_width  = int ( template_width * scale )
        resized_height = int ( template_height * scale )


    # just reset our variables if we didn't need to scale up our original image
    else:
        scale = 1.0
        resized_width  = template_width
        resized_height = template_height
    

    # now we look through the image by scaling down the template
    # since we're scaling down, we don't have to worry about the og's dimension limits.
    # so instead we set the limit of how many bad matches we get in succession and
    # make sure our scale doesnt go below 0
    # same code as our first loop except we increment our scale instead of decrementing

    xBoundary = img_width * 0.1
    yBoundary = img_height * 0.1
    
    while resized_width > xBoundary and resized_height > yBoundary:
        resized_gray = cv2.resize(template_gray, (resized_width, resized_height))

        result = cv2.matchTemplate(img_gray, resized_gray, cv2.TM_CCORR_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)
 
        if max_val > best_match:
            best_match = max_val
            best_location = max_loc
            best_width = resized_width
            best_height = resized_height
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
    print (best_match)
    return subimage, best_width, best_height, best_result




# mean squared error formula/function
def mse(img1, img2):
    # store our image's width and height. they should be the same
    img_width = img1.shape[1]
    img_height = img1.shape[0]

    # subtract our image. will return a 2d array of 0's and 1's
    # 0 should represent if the pixels are the same. 1 if not
    diff_img = cv2.subtract(img1, img2)

    # sum up all the diff squared
    error = np.sum(diff_img**2)

    # calculate the mse
    mse = error/(float(img_height*img_width))
    
    return mse, diff_img




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


#main()
gearBar = cv2.imread('mats.png')
cardbg = cv2.imread("Card_Item_Bg_N.png", cv2.IMREAD_UNCHANGED)
wolfimg = cv2.imread("Item_Icon_Material_Wolfsegg_0.png", cv2.IMREAD_UNCHANGED)

#added_image = cv2.addWeighted(cardbg,0.4,wolfimg,0.1,0)
gearBar = cv2.imread("owned amount.png")
#cv2.imwrite('added_image', added_image)

#wolfsub, _, _, _ = subimageRGBSearch(gearBar, added_image)
#cv2.imshow("wolfsub", wolfsub)

#cv2.imshow("gearBar", gearBar)
#gearBar = cv2.cvtColor(gearBar, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gearBar", gearBar)

#added_image = cv2.addWeighted(cardbg,0.4,wolfimg,0.1,0)
#cv2.imwrite('combined.png', added_image)

path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = path_to_tesseract
text = pytesseract.image_to_string(gearBar)
print(text)
