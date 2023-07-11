import os
import cv2



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
GEAR_X_OFFSET = 4
GEAR_Y_OFFSET = 6

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

SKILL_SLOT_STAR_REQUIREMENTS = [1, 1, 2, 3]
UE_SLOT_STAR_REQUIREMENT = 5
# array of what min level you need to be to unlock the respective gear slot. slot1=lvl0, slot2=lvl15, slot3=lvl35
GEAR_SLOT_LEVEL_REQUIREMENTS = [0, 15, 35]



# STATS templates and masks
STATS_TEMPLATE_IMAGE = cv2.imread("stats template.png", cv2.IMREAD_COLOR)
STATS_MASK_IMAGE = cv2.imread("stats mask.png", cv2.IMREAD_GRAYSCALE)

STATS_NAME_MASK_IMAGE = cv2.imread("stats name mask.png", cv2.IMREAD_GRAYSCALE)
STATS_BOND_MASK_IMAGE = cv2.imread("stats bond mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_MASK_IMAGE = cv2.imread("stats level mask.png", cv2.IMREAD_GRAYSCALE)

NAME_ADDITION_IMAGE = cv2.imread("name addition.png", cv2.IMREAD_GRAYSCALE)

BOND_LEVEL_TEMPLATE_IMAGES = []
BOND_LEVEL_0_TEMPLATE_IMAGE = cv2.imread("bond level 0 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_1_TEMPLATE_IMAGE = cv2.imread("bond level 1 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_2_TEMPLATE_IMAGE = cv2.imread("bond level 2 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_3_TEMPLATE_IMAGE = cv2.imread("bond level 3 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_4_TEMPLATE_IMAGE = cv2.imread("bond level 4 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_5_TEMPLATE_IMAGE = cv2.imread("bond level 5 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_6_TEMPLATE_IMAGE = cv2.imread("bond level 6 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_7_TEMPLATE_IMAGE = cv2.imread("bond level 7 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_8_TEMPLATE_IMAGE = cv2.imread("bond level 8 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_9_TEMPLATE_IMAGE = cv2.imread("bond level 9 template.png", cv2.IMREAD_COLOR)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_0_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_1_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_2_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_3_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_4_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_5_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_6_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_7_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_8_TEMPLATE_IMAGE)
BOND_LEVEL_TEMPLATE_IMAGES.append(BOND_LEVEL_9_TEMPLATE_IMAGE)

BOND_LEVEL_MASK_IMAGES = []
BOND_LEVEL_0_MASK_IMAGE = cv2.imread("bond level 0 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_1_MASK_IMAGE = cv2.imread("bond level 1 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_2_MASK_IMAGE = cv2.imread("bond level 2 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_3_MASK_IMAGE = cv2.imread("bond level 3 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_4_MASK_IMAGE = cv2.imread("bond level 4 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_5_MASK_IMAGE = cv2.imread("bond level 5 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_6_MASK_IMAGE = cv2.imread("bond level 6 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_7_MASK_IMAGE = cv2.imread("bond level 7 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_8_MASK_IMAGE = cv2.imread("bond level 8 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_9_MASK_IMAGE = cv2.imread("bond level 9 mask.png", cv2.IMREAD_GRAYSCALE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_0_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_1_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_2_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_3_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_4_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_5_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_6_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_7_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_8_MASK_IMAGE)
BOND_LEVEL_MASK_IMAGES.append(BOND_LEVEL_9_MASK_IMAGE)


STATS_LEVEL_TEMPLATE_IMAGES = []
STATS_LEVEL_0_TEMPLATE_IMAGE = cv2.imread("stats level 0 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_1_TEMPLATE_IMAGE = cv2.imread("stats level 1 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_2_TEMPLATE_IMAGE = cv2.imread("stats level 2 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_3_TEMPLATE_IMAGE = cv2.imread("stats level 3 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_4_TEMPLATE_IMAGE = cv2.imread("stats level 4 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_5_TEMPLATE_IMAGE = cv2.imread("stats level 5 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_6_TEMPLATE_IMAGE = cv2.imread("stats level 6 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_7_TEMPLATE_IMAGE = cv2.imread("stats level 7 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_8_TEMPLATE_IMAGE = cv2.imread("stats level 8 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_9_TEMPLATE_IMAGE = cv2.imread("stats level 9 template.png", cv2.IMREAD_COLOR)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_0_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_1_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_2_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_3_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_4_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_5_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_6_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_7_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_8_TEMPLATE_IMAGE)
STATS_LEVEL_TEMPLATE_IMAGES.append(STATS_LEVEL_9_TEMPLATE_IMAGE)

STATS_LEVEL_MASK_IMAGES = []
STATS_LEVEL_0_MASK_IMAGE = cv2.imread("stats level 0 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_1_MASK_IMAGE = cv2.imread("stats level 1 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_2_MASK_IMAGE = cv2.imread("stats level 2 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_3_MASK_IMAGE = cv2.imread("stats level 3 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_4_MASK_IMAGE = cv2.imread("stats level 4 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_5_MASK_IMAGE = cv2.imread("stats level 5 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_6_MASK_IMAGE = cv2.imread("stats level 6 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_7_MASK_IMAGE = cv2.imread("stats level 7 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_8_MASK_IMAGE = cv2.imread("stats level 8 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_9_MASK_IMAGE = cv2.imread("stats level 9 mask.png", cv2.IMREAD_GRAYSCALE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_0_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_1_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_2_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_3_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_4_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_5_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_6_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_7_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_8_MASK_IMAGE)
STATS_LEVEL_MASK_IMAGES.append(STATS_LEVEL_9_MASK_IMAGE)


STATS_STAR_MASK_IMAGE = cv2.imread("stats star mask.png", cv2.IMREAD_GRAYSCALE)
STAR_TEMPLATE_IMAGE = cv2.imread("star template.png", cv2.IMREAD_COLOR)
STAR_MASK_IMAGE = cv2.imread("star mask.png", cv2.IMREAD_GRAYSCALE)


# EQUIPMENT templates and masks
EQUIPMENT_TEMPLATE_IMAGE = cv2.imread("equipment template.png", cv2.IMREAD_COLOR)
EQUIPMENT_MASK_IMAGE = cv2.imread("equipment mask.png", cv2.IMREAD_GRAYSCALE)


# SKILLS templates and masks
SKILLS_TEMPLATE_IMAGE = cv2.imread("skills template.png", cv2.IMREAD_COLOR)
SKILLS_MASK_IMAGES = []
SKILLS_1_STAR_MASK_IMAGE = cv2.imread("skills 1 star mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_2_STAR_MASK_IMAGE = cv2.imread("skills 2 star mask.png", cv2.IMREAD_GRAYSCALE)
SKILLS_MASK_IMAGES.append(SKILLS_1_STAR_MASK_IMAGE)
SKILLS_MASK_IMAGES.append(SKILLS_2_STAR_MASK_IMAGE)
SKILLS_MASK_IMAGES.append(None)
SKILLS_MASK_IMAGES.append(None)
SKILLS_MASK_IMAGES.append(None)

SKILL_SLOT_MASK_IMAGES = []
SKILL_SLOT_1_MASK_IMAGE = cv2.imread("skill slot 1 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_SLOT_2_MASK_IMAGE = cv2.imread("skill slot 2 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_SLOT_3_MASK_IMAGE = cv2.imread("skill slot 3 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_SLOT_4_MASK_IMAGE = cv2.imread("skill slot 4 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_SLOT_MASK_IMAGES.append(SKILL_SLOT_1_MASK_IMAGE)
SKILL_SLOT_MASK_IMAGES.append(SKILL_SLOT_2_MASK_IMAGE)
SKILL_SLOT_MASK_IMAGES.append(SKILL_SLOT_3_MASK_IMAGE)
SKILL_SLOT_MASK_IMAGES.append(SKILL_SLOT_4_MASK_IMAGE)

SKILL_LEVEL_COUNT = [5, 10, 10, 10]

SKILL_LEVEL_TEMPLATE_IMAGES = []
SKILL_LEVEL_MAX_TEMPLATE_IMAGE = cv2.imread("skill level max template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_1_TEMPLATE_IMAGE = cv2.imread("skill level 1 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_2_TEMPLATE_IMAGE = cv2.imread("skill level 2 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_3_TEMPLATE_IMAGE = cv2.imread("skill level 3 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_4_TEMPLATE_IMAGE = cv2.imread("skill level 4 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_5_TEMPLATE_IMAGE = cv2.imread("skill level 5 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_6_TEMPLATE_IMAGE = cv2.imread("skill level 6 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_7_TEMPLATE_IMAGE = cv2.imread("skill level 7 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_8_TEMPLATE_IMAGE = cv2.imread("skill level 8 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_9_TEMPLATE_IMAGE = cv2.imread("skill level 9 template.png", cv2.IMREAD_COLOR)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_MAX_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_1_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_2_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_3_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_4_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_5_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_6_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_7_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_8_TEMPLATE_IMAGE)
SKILL_LEVEL_TEMPLATE_IMAGES.append(SKILL_LEVEL_9_TEMPLATE_IMAGE)

SKILL_LEVEL_MASK_IMAGES = []
SKILL_LEVEL_MAX_MASK_IMAGE = cv2.imread("skill level max mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_1_MASK_IMAGE = cv2.imread("skill level 1 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_2_MASK_IMAGE = cv2.imread("skill level 2 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_3_MASK_IMAGE = cv2.imread("skill level 3 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_4_MASK_IMAGE = cv2.imread("skill level 4 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_5_MASK_IMAGE = cv2.imread("skill level 5 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_6_MASK_IMAGE = cv2.imread("skill level 6 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_7_MASK_IMAGE = cv2.imread("skill level 7 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_8_MASK_IMAGE = cv2.imread("skill level 8 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_9_MASK_IMAGE = cv2.imread("skill level 9 mask.png", cv2.IMREAD_GRAYSCALE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_MAX_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_1_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_2_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_3_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_4_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_5_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_6_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_7_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_8_MASK_IMAGE)
SKILL_LEVEL_MASK_IMAGES.append(SKILL_LEVEL_9_MASK_IMAGE)

##SKILL_LEVEL_MAX_TEMPLATE_IMAGE = cv2.imread("skill level max template.png", cv2.IMREAD_COLOR)
##SKILL_LEVEL_MAX_MASK_IMAGE = cv2.imread("skill level max mask.png", cv2.IMREAD_GRAYSCALE)


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

UE_LEVEL_TEMPLATE_IMAGES = []
UE_LEVEL_0_TEMPLATE_IMAGE = cv2.imread("ue level 0 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_1_TEMPLATE_IMAGE = cv2.imread("ue level 1 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_2_TEMPLATE_IMAGE = cv2.imread("ue level 2 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_3_TEMPLATE_IMAGE = cv2.imread("ue level 3 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_4_TEMPLATE_IMAGE = cv2.imread("ue level 4 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_5_TEMPLATE_IMAGE = cv2.imread("ue level 5 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_6_TEMPLATE_IMAGE = cv2.imread("ue level 6 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_7_TEMPLATE_IMAGE = cv2.imread("ue level 7 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_8_TEMPLATE_IMAGE = cv2.imread("ue level 8 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_9_TEMPLATE_IMAGE = cv2.imread("ue level 9 template.png", cv2.IMREAD_COLOR)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_0_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_1_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_2_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_3_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_4_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_5_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_6_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_7_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_8_TEMPLATE_IMAGE)
UE_LEVEL_TEMPLATE_IMAGES.append(UE_LEVEL_9_TEMPLATE_IMAGE)

UE_LEVEL_MASK_IMAGES = []
UE_LEVEL_0_MASK_IMAGE = cv2.imread("ue level 0 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_1_MASK_IMAGE = cv2.imread("ue level 1 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_2_MASK_IMAGE = cv2.imread("ue level 2 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_3_MASK_IMAGE = cv2.imread("ue level 3 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_4_MASK_IMAGE = cv2.imread("ue level 4 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_5_MASK_IMAGE = cv2.imread("ue level 5 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_6_MASK_IMAGE = cv2.imread("ue level 6 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_7_MASK_IMAGE = cv2.imread("ue level 7 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_8_MASK_IMAGE = cv2.imread("ue level 8 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_9_MASK_IMAGE = cv2.imread("ue level 9 mask.png", cv2.IMREAD_GRAYSCALE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_0_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_1_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_2_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_3_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_4_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_5_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_6_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_7_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_8_MASK_IMAGE)
UE_LEVEL_MASK_IMAGES.append(UE_LEVEL_9_MASK_IMAGE)



# GEARS tempaltes and masks
GEARS_TEMPLATE_IMAGE = cv2.imread("gears template.png", cv2.IMREAD_COLOR)
GEARS_MASK_IMAGE = cv2.imread("gears mask.png", cv2.IMREAD_GRAYSCALE)

GEAR_SLOT_MASK_IMAGES = []
GEAR_SLOT_1_MASK_IMAGE = cv2.imread("gear slot 1 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_SLOT_2_MASK_IMAGE = cv2.imread("gear slot 2 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_SLOT_3_MASK_IMAGE = cv2.imread("gear slot 3 mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_SLOT_MASK_IMAGES.append(GEAR_SLOT_1_MASK_IMAGE)
GEAR_SLOT_MASK_IMAGES.append(GEAR_SLOT_2_MASK_IMAGE)
GEAR_SLOT_MASK_IMAGES.append(GEAR_SLOT_3_MASK_IMAGE)

GEAR_E_MASK_IMAGE = cv2.imread("gear E mask.png", cv2.IMREAD_GRAYSCALE)
GEAR_TIER_MASK_IMAGE = cv2.imread("gear tier mask.png", cv2.IMREAD_GRAYSCALE)

E_TEMPLATE_IMAGE = cv2.imread("E template.png", cv2.IMREAD_COLOR)
E_MASK_IMAGE = cv2.imread("E mask.png", cv2.IMREAD_GRAYSCALE)

TIER_LEVEL_TEMPLATE_IMAGES_DIRECTORY_PATH = r"templates and masks\level\skill\masks"
TIER_LEVEL_TEMPLATE_IMAGES = []
TIER_1_TEMPLATE_IMAGE = cv2.imread("tier 1 template.png", cv2.IMREAD_COLOR)
TIER_2_TEMPLATE_IMAGE = cv2.imread("tier 2 template.png", cv2.IMREAD_COLOR)
TIER_3_TEMPLATE_IMAGE = cv2.imread("tier 3 template.png", cv2.IMREAD_COLOR)
TIER_4_TEMPLATE_IMAGE = cv2.imread("tier 4 template.png", cv2.IMREAD_COLOR)
TIER_5_TEMPLATE_IMAGE = cv2.imread("tier 5 template.png", cv2.IMREAD_COLOR)
TIER_6_TEMPLATE_IMAGE = cv2.imread("tier 6 template.png", cv2.IMREAD_COLOR)
TIER_7_TEMPLATE_IMAGE = cv2.imread("tier 7 template.png", cv2.IMREAD_COLOR)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_1_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_2_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_3_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_4_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_5_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_6_TEMPLATE_IMAGE)
TIER_LEVEL_TEMPLATE_IMAGES.append(TIER_7_TEMPLATE_IMAGE)
TIER_LEVEL_MASK_IMAGE = cv2.imread("tier level mask.png", cv2.IMREAD_GRAYSCALE)

##def readImagesToArray(directory):
##    array = []
##    
##    if
##    
##    for fileName in os.listdir(directory):
##        filePath = os.path.join(directory, fileName)
##        
##        if "template" in fileName:
##            image = cv2.imread(filePath, cv2.IMREAD_COLOR)
##        else:
##            image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
##            
##        array.append(image)
##
##    return array
##directory = r"templates and masks\levels\bond\templates"
##directory = r"templates and masks\levels\bond\masks"
##directory = r"templates and masks\levels\gear\templates"
##directory = r"templates and masks\levels\gear\masks"
##directory = r"templates and masks\levels\skill\templates"
##directory = r"templates and masks\levels\skill\masks"
##directory = r"templates and masks\levels\stats\templates"
##directory = r"templates and masks\levels\stats\masks"
##directory = r"templates and masks\levels\tier\templates"
##directory = r"templates and masks\levels\tier\masks"
##directory = r"templates and masks\levels\ue\templates"
##directory = r"templates and masks\levels\ue\masks"

##readImagesToArray(directory)
