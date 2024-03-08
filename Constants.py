import os
import cv2
import numpy as np



SCALE_CHEAT_SHEET = [ 1.0, 0.8999999999999999, 1.0, 0.8999999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ,1.0, 1.0, 1.0, 1.0, \
                      0.8899999999999999, 0.8899999999999999, 0.8899999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                      1.0, 1.0, 0.8899999999999999, 0.5086363636363634, 0.8899999999999999, 1.0, 1.0, 1.0, 1.0, \
                      0.8899999999999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


NAME_CHEAT_SHEET = ["Airi", "Airi", "Akane (Bunny)", "Akane", "Ako","Asuna","Ayane","Ayane (Swimsuit)","Cherino","Chihiro","Eimi","Hanae","Hanako","Hare","Hasumi (Track)",\
                    "Hibiki (Cheer Squad)","Hibiki (Cheer Squad)","Hibiki (Cheer Squad)","Hibiki (Cheer Squad)","Himari","Hinata","Iori","Izumi","Junko","Juri","Kirino","Maki","Maki",\
                    "Marina","Mashiro","Michiru","Midori","Hatsune Miku","Mutsuki","Natsu","Neru","Nodoka","Chinatsu (Hot Spring)","Saya","Shimiko","Shizuko", "Shun (Small)",\
                    "Izumi (Swimsuit)", "Sumire","Suzumi","Tsurugi","Ui","Yoshimi","Yuuka (Track)"]

STAR_CHEAT_SHEET = [5,4,3,5,5,5,3,3,5,3,5,5,4,4,5,2,1,1,1,3,3,5,3,4,4,3,5,5,3,5,4,5,3,5,3,3,4,3,4,5,5,3,4,5,4,4,3,5,3]

BOND_CHEAT_SHEET_ARRAY = []
BOND_CHEAT_SHEET = [20,20,9,20,21,21,20,14,22,16,20,22,19,19,6,6,2,2,1,3,13,22,20,20,20,18,24,24,8,17,14,19,20,23,20,8,20,12,20,20,17,20,20,20,18,15,17,20,4]
for bond in BOND_CHEAT_SHEET:
    bondArray = [int(x) for x in str(bond)]
    BOND_CHEAT_SHEET_ARRAY.append(bondArray)


LEVEL_CHEAT_SHEET = [1,1,1,80,80,75,9,3,80,35,75,75,1,5,80,1,1,1,1,75,75,80,10,71,7,38,80,80,1,6,15,75,1,80,75,76,45,70,\
                     35,8,75,2,35,12,4,1,75,56,75]
LEVEL_CHEAT_SHEET_ARRAY = []
for level in LEVEL_CHEAT_SHEET:
    levelArray = [int(x) for x in str(level)]
    LEVEL_CHEAT_SHEET_ARRAY.append(levelArray)

SKILL_LEVEL_CHEAT_SHEET = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[5,1,7,1],[5,10,10,10],[4,5,7,7],[1,1,1,1],[1,1,1,1],[5,10,10,10],[3,1,1,7],[3,2,4,3],\
                           [5,4,7,7],[1,1,1,1],[1,1,1,1],[5,4,10,7],[1,1,1,0],[1,1,0,0],[1,1,0,0],[1,1,0,0],[5,4,4,7],[3,1,4,4],[5,10,4,10],[1,1,1,1],[4,4,4,7],\
                           [1,1,1,1],[1,1,1,1],[5,10,10,10],[5,10,10,10],[1,1,1,1],[3,1,1,7],[1,1,1,1],[5,4,8,9],[1,1,1,1],[5,10,10,7],[4,7,6,7],\
                           [1,1,1,1],[5,7,1,7],[5,10,1,1],[1,1,1,1],[1,1,1,1],[4,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[4,1,1,1],[1,1,1,1],[5,7,7,7],\
                           [2,1,1,1],[5,10,7,7]]

UE_STAR_CHEAT_SHEET = [1,0,0,2,2,3,0,0,2,0,2,2,0,0,3,0,0,0,0,0,0,2,0,0,0,0,3,3,0,3,0,1,0,3,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0]

UE_LEVEL_CHEAT_SHEAT = [1,0,0,40,40,50,0,0,40,0,40,40,0,0,50,0,0,0,0,0,0,40,0,0,0,0,50,50,0,50,0,30,0,50,0,0,0,0,0,30,0,0,0,30,0,0,0,30,0]
UE_LEVEL_CHEAT_SHEAT_ARRAY = []
for ueLevel in UE_LEVEL_CHEAT_SHEAT:
    ueLevelArray = [int(x) for x in str(ueLevel)]
    UE_LEVEL_CHEAT_SHEAT_ARRAY.append(ueLevelArray)

GEAR_TIERS_CHEAT_SHEET_ARRAY = [[0,0,0],[0,0,0],[0,0,0],[7,6,5],[7,7,6],[7,6,6],[0,0,0],[0,0,0],[7,7,5],[1,1,6],[7,7,6],[2,2,5],[0,0,0],[0,0,0],[7,3,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],\
                          [6,1,4],[7,6,7],[0,0,0],[5,2,3],[0,0,0],[5,5,5],[7,6,6],[7,6,6],[0,0,0],[0,0,0],[1,0,0],[7,6,6],[0,0,0],[7,6,6],[7,7,6],[5,6,6],[2,1,2],[6,3,1],[1,1,1],\
                          [0,0,0],[7,6,5],[0,0,0],[4,3,5],[0,0,0],[0,0,0],[0,0,0],[7,6,6],[1,1,1],[7,7,7]]







PATH_TO_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
##print(os.getcwd())
os.chdir(r"templates and masks")
##print(os.getcwd())

BAD_COUNTER_MAX = 150000
SCALE_INCREMENT = 0.01



## recorded the highest correct guess and lowest wrong guess
# min miss: 0.04736049473285675
# max hit: 0.038083869963884354
BOND_LEVEL_MATCH_THRESHOLD = 0.0427

## recorded the highest value in overlapPercentages of values that were below 0.50 and the lowest value of those that were above 0.50
## should remember that overlap is determined by the match threshold because match threshold is the initial filter
# min miss: 0.9212121212121213
# max hit: 0.4090909090909091
BOND_LEVEL_OVERLAP_THRESHOLD = 0.665


## same process as bond_match.
## even with the match threshold being high, the nms should filter out the bad eggs... hopefully...
# min miss: 0.060845356434583664
# max hit: 0.061924394220113754
STATS_LEVEL_MATCH_THRESHOLD = 0.07

# same process and bond_overlap
# min miss: 0.8625
# max hit: 0.46153846153846156
STATS_LEVEL_OVERLAP_THRESHOLD = 0.662


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

# min miss: 0.047349803149700165
# max hit: 0.04790861904621124
UE_LEVEL_MATCH_THRESHOLD = 0.055

# max hit: 0.25
UE_LEVEL_OVERLAP_THRESHOLD = 0.3


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




# REQUIREMENTS
# skills' character star requirement to unlock respective skill. index0 represents skill1, .... so far only 4 skills
SKILLS_SLOT_STAR_REQUIREMENTS = [1, 1, 2, 3]
SKILLS_SLOT_LEVEL_MAX = [5, 10, 10, 10]

# ue's character star requirement to unlock it. currently need 5 star. *** debating whether or not i should
# make it go up to 7 star, for each upgrade in the ue's star.***
UE_SLOT_STAR_REQUIREMENT = 5

# gears' character level requirement to unlock the respective gear. slot1=lvl0, slot2=lvl15, slot3=lvl35
GEARS_SLOT_LEVEL_REQUIREMENTS = [0, 15, 35]






# level templates and masks paths
LEVELS_DIRECTORY_PATH = r"levels"
BOND_LEVELS_DIRECTORY_PATH =  LEVELS_DIRECTORY_PATH + r"\bond"
GEAR_LEVELS_DIRECTORY_PATH =  LEVELS_DIRECTORY_PATH + r"\gear"
SKILL_LEVELS_DIRECTORY_PATH = LEVELS_DIRECTORY_PATH + r"\skill"
STATS_LEVELS_DIRECTORY_PATH = LEVELS_DIRECTORY_PATH + r"\stats"
TIER_LEVELS_DIRECTORY_PATH =  LEVELS_DIRECTORY_PATH + r"\tier"
UE_LEVELS_DIRECTORY_PATH =    LEVELS_DIRECTORY_PATH + r"\ue"



# STATS templates and masks
STATS_TEMPLATE_IMAGE = cv2.imread("stats template.png", cv2.IMREAD_COLOR)
STATS_MASK_IMAGE = cv2.imread("stats mask.png", cv2.IMREAD_GRAYSCALE)

STATS_NAME_MASK_IMAGE = cv2.imread("stats name mask.png", cv2.IMREAD_GRAYSCALE)
STATS_BOND_MASK_IMAGE = cv2.imread("stats bond mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_MASK_IMAGE = cv2.imread("stats level mask.png", cv2.IMREAD_GRAYSCALE)
STATS_STAR_MASK_IMAGE = cv2.imread("stats star mask.png", cv2.IMREAD_GRAYSCALE)

NAME_ADDITION_IMAGE = cv2.imread("name addition.png", cv2.IMREAD_GRAYSCALE)

STAR_TEMPLATE_IMAGE = cv2.imread("star template.png", cv2.IMREAD_COLOR)
STAR_MASK_IMAGE = cv2.imread("star mask.png", cv2.IMREAD_GRAYSCALE)


# EQUIPMENT templates and masks
EQUIPMENT_TEMPLATE_IMAGE = cv2.imread("equipment template.png", cv2.IMREAD_COLOR)
EQUIPMENT_MASK_IMAGE = cv2.imread("equipment mask.png", cv2.IMREAD_GRAYSCALE)


# SKILLS templates and masks
SKILLS_TEMPLATE_IMAGE = cv2.imread("skills template.png", cv2.IMREAD_COLOR)


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

GEAR_E_MASK_IMAGE = cv2.imread("gear E mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_TIER_MASK_IMAGE = cv2.imread("gear tier mask.png", cv2.IMREAD_GRAYSCALE)

E_TEMPLATE_IMAGE = cv2.imread("E template.png", cv2.IMREAD_COLOR)
E_MASK_IMAGE = cv2.imread("E mask.png", cv2.IMREAD_GRAYSCALE)




def readLevelsImagesToArrays(levelsDirectory):
    levelsTemplatesDirectory = levelsDirectory + r"\templates"
    levelsMasksDirectory = levelsDirectory + r"\masks"
    
    templatesArray = readLevelsImagesToArray(levelsTemplatesDirectory, cv2.IMREAD_COLOR)
    masksArray = readLevelsImagesToArray(levelsMasksDirectory, cv2.IMREAD_GRAYSCALE)
    

    return templatesArray, masksArray


def readLevelsImagesToArray(levelsDirectory, imReadColor):
    imagesArray = []
    
    for fileName in os.listdir(levelsDirectory):
        
        filePath = os.path.join(levelsDirectory, fileName)
        
        image = cv2.imread(filePath, imReadColor)
            
        imagesArray.append(image)

    return imagesArray


BOND_LEVEL_TEMPLATE_IMAGES, BOND_LEVEL_MASK_IMAGES = readLevelsImagesToArrays(BOND_LEVELS_DIRECTORY_PATH)
GEAR_LEVEL_TEMPLATE_IMAGES, GEAR_LEVEL_MASK_IMAGES = readLevelsImagesToArrays(GEAR_LEVELS_DIRECTORY_PATH)
SKILL_LEVEL_TEMPLATE_IMAGES, SKILL_LEVEL_MASK_IMAGES = readLevelsImagesToArrays(SKILL_LEVELS_DIRECTORY_PATH)
STATS_LEVEL_TEMPLATE_IMAGES, STATS_LEVEL_MASK_IMAGES = readLevelsImagesToArrays(STATS_LEVELS_DIRECTORY_PATH)
UE_LEVEL_TEMPLATE_IMAGES, UE_LEVEL_MASK_IMAGES = readLevelsImagesToArrays(UE_LEVELS_DIRECTORY_PATH)

TIER_LEVEL_TEMPLATES_DIRECTORY_PATH = TIER_LEVELS_DIRECTORY_PATH  + r"\templates"
TIER_LEVEL_TEMPLATE_IMAGES = readLevelsImagesToArray(TIER_LEVEL_TEMPLATES_DIRECTORY_PATH, cv2.IMREAD_COLOR)
TIER_LEVEL_MASK_IMAGE = cv2.imread(TIER_LEVELS_DIRECTORY_PATH + r"\masks\tier level mask.png", cv2.IMREAD_GRAYSCALE)




GEARS_SLOT_MASK_IMAGES = []
GEARS_SLOT_1_MASK_IMAGE = cv2.imread("gears slot 1 mask.png", cv2.IMREAD_GRAYSCALE)
GEARS_SLOT_2_MASK_IMAGE = cv2.imread("gears slot 2 mask.png", cv2.IMREAD_GRAYSCALE)
GEARS_SLOT_3_MASK_IMAGE = cv2.imread("gears slot 3 mask.png", cv2.IMREAD_GRAYSCALE)
GEARS_SLOT_MASK_IMAGES.append(GEARS_SLOT_1_MASK_IMAGE)
GEARS_SLOT_MASK_IMAGES.append(GEARS_SLOT_2_MASK_IMAGE)
GEARS_SLOT_MASK_IMAGES.append(GEARS_SLOT_3_MASK_IMAGE)

SKILLS_MASK_IMAGES = []
SKILLS_1_STAR_MASK_IMAGE = cv2.imread("skills 1 star mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_2_STAR_MASK_IMAGE = cv2.imread("skills 2 star mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_MASK_IMAGES.append(SKILLS_1_STAR_MASK_IMAGE)
SKILLS_MASK_IMAGES.append(SKILLS_2_STAR_MASK_IMAGE)
SKILLS_MASK_IMAGES.append(None)
SKILLS_MASK_IMAGES.append(None)
SKILLS_MASK_IMAGES.append(None)

SKILLS_LEVEL_SLOT_MASK_IMAGES = []
SKILLS_LEVEL_SLOT_1_MASK_IMAGE = cv2.imread("skills slot 1 mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_LEVEL_SLOT_2_MASK_IMAGE = cv2.imread("skills slot 2 mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_LEVEL_SLOT_3_MASK_IMAGE = cv2.imread("skills slot 3 mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_LEVEL_SLOT_4_MASK_IMAGE = cv2.imread("skills slot 4 mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_LEVEL_SLOT_MASK_IMAGES.append(SKILLS_LEVEL_SLOT_1_MASK_IMAGE)
SKILLS_LEVEL_SLOT_MASK_IMAGES.append(SKILLS_LEVEL_SLOT_2_MASK_IMAGE)
SKILLS_LEVEL_SLOT_MASK_IMAGES.append(SKILLS_LEVEL_SLOT_3_MASK_IMAGE)
SKILLS_LEVEL_SLOT_MASK_IMAGES.append(SKILLS_LEVEL_SLOT_4_MASK_IMAGE)



counter = 0
def printImagesFromArray(imagesArray, counter):
    for image in imagesArray:
        cv2.imshow(str(counter), image)
        counter += 1

    return counter
##
##counter = printImagesFromArray(BOND_LEVEL_TEMPLATE_IMAGES, counter)
##counter = printImagesFromArray(GEAR_LEVEL_TEMPLATE_IMAGES, counter)
##counter = printImagesFromArray(SKILL_LEVEL_TEMPLATE_IMAGES, counter)
##counter = printImagesFromArray(STATS_LEVEL_TEMPLATE_IMAGES, counter)
##counter = printImagesFromArray(TIER_LEVEL_TEMPLATE_IMAGES, counter)
##counter = printImagesFromArray(UE_LEVEL_TEMPLATE_IMAGES, counter)



def compareTwoImageLists(imageList1, imageList2):
    print(len(imageList1))
    for counter in range(len(imageList1)):
        
        cv2.imshow("imageList1 " + str(counter), imageList1[counter])
        cv2.imshow("imageList2 " + str(counter), imageList2[counter])
        
        comparison = imageList1[counter] == imageList2[counter]
        
        if comparison.all():
            print("true")
        else:
            print("false")
