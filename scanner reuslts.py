Python 3.7.2 (tags/v3.7.2:9a3ffc0492, Dec 23 2018, 22:20:52) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi ipad.png
Airi 20 1 5 [1 1 1 1] 1 1 [0 0 0]
totaltime:62.18983244895935
student example\airi.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 937, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 916, in main
    studentSkills = getStudentSkills(equipmentImage, scale, 3)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 737, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi ipad.png
Airi 20 1 5 [1 1 1 1] 1 1 [0 0 0]
totaltime:62.260077476501465
student example\airi.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 937, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 916, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 737, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 937, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 916, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 737, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Airi 20 1 4
[[[244 228 206]
  [243 233 214]
  [244 234 215]
  ...
  [198 188 176]
  [189 180 172]
  [222 216 207]]

 [[244 228 206]
  [243 233 214]
  [244 234 215]
  ...
  [198 189 176]
  [189 181 173]
  [222 217 209]]

 [[244 229 206]
  [243 233 214]
  [244 234 215]
  ...
  [198 190 177]
  [189 182 175]
  [223 219 211]]

 ...

 [[212 199 133]
  [243 233 214]
  [244 234 215]
  ...
  [173 163 154]
  [138 130 128]
  [159 155 155]]

 [[211 198 132]
  [243 233 214]
  [244 234 215]
  ...
  [175 166 156]
  [141 134 133]
  [163 159 160]]

 [[211 198 130]
  [243 233 214]
  [244 234 215]
  ...
  [177 168 158]
  [145 138 138]
  [167 164 165]]] 0.8999999999999999 [[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]  
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 940, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 919, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 739, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Airi 20 1 4
0.8999999999999999
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 945, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 924, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 744, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Airi 20 1 4
0.8999999999999999
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 945, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 924, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 744, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 647, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> yes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> yes
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> yes[:5]
[0, 1, 2, 3, 4]
>>> SKILL_LEVEL_TEMPLATE_IMAGES

>>> type(SKILL_LEVEL_TEMPLATE_IMAGES)
<class 'list'>
>>> type(SKILL_LEVEL_TEMPLATE_IMAGES[:5])
<class 'list'>
>>> len(SKILL_LEVEL_TEMPLATE_IMAGES)
10
>>> len(SKILL_LEVEL_TEMPLATE_IMAGES[:5])
5
>>> SKILLS_SLOT_LEVEL_MAX
[5, 10, 10, 10]
>>> SKILLS_SLOT_LEVEL_MAX[0]
5
>>> len(SKILL_LEVEL_TEMPLATE_IMAGES)
10
>>> len(SKILL_LEVEL_TEMPLATE_IMAGES[:SKILLS_SLOT_LEVEL_MAX])
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    len(SKILL_LEVEL_TEMPLATE_IMAGES[:SKILLS_SLOT_LEVEL_MAX])
TypeError: slice indices must be integers or None or have an __index__ method
>>> len(SKILL_LEVEL_TEMPLATE_IMAGES[:SKILLS_SLOT_LEVEL_MAX[0]])
5
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
levelCount 10
levelCount 10
Airi 20 1 4
0.8999999999999999
levelCount 5
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 950, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 749, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 652, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
levelCount 10
0.0337824821472168
0.08789713680744171
0.031300146132707596
0.11370545625686646
0.13124825060367584
0.11741414666175842
0.08436685055494308
0.06451702117919922
0.09931284934282303
0.08162885904312134
levelCount 10
0.20289921760559082
0.007242770865559578
0.20753999054431915
0.17663085460662842
0.16210444271564484
0.22283311188220978
0.202458918094635
0.19168728590011597
0.20649993419647217
0.18168719112873077
Airi 20 1 4
0.8999999999999999
levelCount 5
0.15615105628967285
0.011776071041822433
0.12330202758312225
0.11054226011037827
0.11848042160272598
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 950, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    studentSkills = getStudentSkills(equipmentImage, scale, studentStar)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 749, in getStudentSkills
    skillsLevel = getLevels(skillsSubimage, imageScale, SKILLS_LEVEL_SLOT_MASK_IMAGES[skillsSlot], SKILL_LEVEL_TEMPLATE_IMAGES[:skillsSlotLevelMax], SKILL_LEVEL_MASK_IMAGES[:skillsSlotLevelMax], 0.01, 0.665)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 652, in getLevels
    nms2Results, nms2Locations, nmsCount, nms2Order = nms(levelsNMSResults, levelsNMSLocations, levelsWidths, levelsHeights, overlapThreshold)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 126, in nms
    x1MatchCoordinates = matchLocations[:,0]
IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Airi 20 1 4
Airi 20 1 4 [1 1 1 1] 0 0 [0 0 0]
totaltime:23.72864055633545
student example\akane bunny ipad.png
Akane (Bunny) 9 1 3
Akane (Bunny) 9 1 3 [1 1 1 1] 0 0 [0 0 0]
totaltime:62.94606804847717
student example\akane.png
Akane 20 80 5
Akane 20 80 5 [5 1 7 1] 2 40 [7 3 5]
totaltime:22.722482919692993
student example\ako ipad.png
Ako 21 80 5
Ako 21 80 5 [ 5 10 10 10] 2 40 [7 7 6]
totaltime:61.64130973815918
student example\asuna ipad.png
Asuna 21 75 5
Asuna 21 75 5 [4 5 7 7] 3 50 [7 3 3]
totaltime:61.54706835746765
student example\ayane ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 950, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 924, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 466, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Airi 20 1 4 [1 1 1 1] 0 0 [0 0 0]
totaltime:23.13334083557129
student example\akane bunny ipad.png
Akane (Bunny) 9 1 3 [1 1 1 1] 0 0 [0 0 0]
totaltime:62.136611461639404
student example\akane.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 952, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 933, in main
    gearTiers = getStudentGears(equipmentImage, scale, studentLevel)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 865, in getStudentGears
    gearTier = getGearTier(gearSubimage, imageScale)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 817, in getGearTier
    drawBoxes(tierSubimage, tierSubimageResults, tierSubimageWidth, tierSubimageHeight)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 343, in drawBoxes
    cv2.rectangle(sourceImage, (location), (x2[index], y2[index]), colorr, 1)
cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Can't parse 'pt1'. Expected sequence length 2, got 17
>  - Can't parse 'pt1'. Expected sequence length 2, got 17
>  - Can't parse 'rec'. Expected sequence length 4, got 17
>  - Can't parse 'rec'. Expected sequence length 4, got 17

>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\airi.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 953, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 927, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 466, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Akane 20 80 5 [5 1 7 1] 2 40 [7 3 5]
totaltime:22.800379753112793
student example\asuna ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 953, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 927, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 466, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Akane 20 80 5 [5 1 7 1] 2 40 [7 3 5]
totaltime:23.02011775970459
student example\asuna ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 954, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 928, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 466, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 954, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 935, in main
    gearTiers = getStudentGears(equipmentImage, scale, studentLevel)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 867, in getStudentGears
    gearTier = getGearTier(gearSubimage, imageScale)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 823, in getGearTier
    cv2.imshow(str(time.time() + "tierSubimage" + str(tier)), tierSubimage)
TypeError: unsupported operand type(s) for +: 'float' and 'str'
>>> 
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Akane 20 80 5 [5 1 7 1] 2 40 [7 3 5]
totaltime:22.897409677505493
student example\asuna ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 954, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 928, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 466, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 954, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 935, in main
    gearTiers = getStudentGears(equipmentImage, scale, studentLevel)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 867, in getStudentGears
    gearTier = getGearTier(gearSubimage, imageScale)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 818, in getGearTier
    drawBoxes(gearTierSubimage, tierSubimageResults, tierSubimageWidth, tierSubimageHeight)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 343, in drawBoxes
    cv2.rectangle(sourceImage, (location), (x2[index], y2[index]), colorr, 1)
cv2.error: OpenCV(4.7.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Can't parse 'pt1'. Expected sequence length 2, got 17
>  - Can't parse 'pt1'. Expected sequence length 2, got 17
>  - Can't parse 'rec'. Expected sequence length 4, got 17
>  - Can't parse 'rec'. Expected sequence length 4, got 17

>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
tier0:0.023410877212882042
tier1:0.019243551418185234
tier2:0.0339592806994915
tier3:0.035660192370414734
tier4:0.029716303572058678
tier5:0.03703545406460762
tier6:0.011963872238993645
tier0:0.07899536192417145
tier1:0.06215681508183479
tier2:0.045096687972545624
tier3:0.05305291712284088
tier4:0.05077386274933815
tier5:0.058843132108449936
tier6:0.07237964123487473
tier0:0.025381112471222878
tier1:0.024434659630060196
tier2:0.02316305786371231
tier3:0.02701941877603531
tier4:0.0077071161940693855
tier5:0.01591743715107441
tier6:0.030087128281593323
Akane 20 80 5 [5 1 7 1] 2 40 [7 3 5]
totaltime:23.469366550445557
student example\asuna ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 955, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 468, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
tier0:0.02671913430094719
tier1:0.021282121539115906
tier2:0.02778792381286621
tier3:0.04206918552517891
tier4:0.03570254147052765
tier5:0.044571083039045334
tier6:0.011558251455426216
tier0:0.04909365996718407
tier1:0.0411626361310482
tier2:0.03497549146413803
tier3:0.0339212641119957
tier4:0.026142369955778122
tier5:0.02172900177538395
tier6:0.05468842387199402
tier0:0.031187891960144043
tier1:0.03034992143511772
tier2:0.01737838052213192
tier3:0.033370260149240494
tier4:0.00948239304125309
tier5:0.018985461443662643
tier6:0.037576623260974884
Akane 20 80 5 [5 1 7 1] 2 40 [7 6 5]
totaltime:23.57125186920166
student example\asuna ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 955, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 468, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 955, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 468, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png

 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
student example\akane.png
Akane 20 80 5 [5 1 7 1] 2 40 [7 6 5]
totaltime:22.786367893218994
student example\asuna ipad.png
Asuna 21 75 5 [4 5 7 7] 3 50 [7 6 6]
totaltime:61.77960443496704
student example\ayane ipad.png
Traceback (most recent call last):
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 955, in <module>
    studentName, studentBond, studentLevel, studentStar, studentSkills, ueStar, ueLevel, gearTiers = main(sourceImage)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 929, in main
    equipmentImage, _, _, scale, _ = subimageMultiScaleSearch(sourceImage, EQUIPMENT_TEMPLATE_IMAGE, EQUIPMENT_MASK_IMAGE)
  File "C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py", line 468, in subimageMultiScaleSearch
    matchResults = cv2.matchTemplate(graySourceImage, scaledGrayTemplateImage, TEMPLATE_MATCH_METHOD, mask = scaledMaskImage)
KeyboardInterrupt
>>> 
 RESTART: C:\Users\appsm\IdeaProjects\Blue Archive Character Logger\ba test scanner.py 
1 hello 2
student example\airi ipad.png
Airi bond  20 , level 1 5 star [1 1 1 1] 1 1 [0 0 0]
totaltime:67.75194001197815
student example\airi.png
Airi bond  20 , level 1 4 star [1 1 1 1] 0 0 [0 0 0]
totaltime:25.379172325134277
student example\akane bunny ipad.png
Akane (Bunny) bond  9 , level 1 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:67.44465827941895
student example\akane.png
Akane bond  20 , level 80 5 star [5 1 7 1] 2 40 [7 6 5]
totaltime:24.68784236907959
student example\ako ipad.png
Ako bond  21 , level 80 5 star [ 5 10 10 10] 2 40 [7 7 6]
totaltime:67.01051378250122
student example\asuna ipad.png
Asuna bond  21 , level 75 5 star [4 5 7 7] 3 50 [7 6 6]
totaltime:67.02441644668579
student example\ayane ipad.png
Ayane bond  20 , level 9 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:66.83582544326782
student example\ayane ss ipad.png
Ayane (Swimsuit) bond  14 , level 3 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:66.96280193328857
student example\cherino ipad.png
Cherino bond  22 , level 80 5 star [ 5 10 10 10] 2 40 [7 7 5]
totaltime:66.67413902282715
student example\chihiro ipad.png
Chihiro bond  16 , level 35 3 star [3 1 1 7] 0 0 [1 1 6]
totaltime:66.86061358451843
student example\eimi ipad.png
Eimi bond  20 , level 75 5 star [3 2 4 3] 2 40 [7 7 6]
totaltime:66.210284948349
student example\hanae ipad.png
Hanae bond  22 , level 75 5 star [5 4 7 7] 2 40 [2 2 5]
totaltime:66.34638118743896
student example\hanako ipad.png
Hanako bond  19 , level 1 4 star [1 1 1 1] 0 0 [0 0 0]
totaltime:66.51383256912231
student example\hare ipad.png
Hare bond  19 , level 5 4 star [1 1 1 1] 0 0 [0 0 0]
totaltime:66.53469014167786
student example\hasumi ipad.png
Hasumi (Track) bond  6 , level 80 5 star [ 5  4 10  7] 3 50 [7 3 1]
totaltime:66.16238641738892
student example\hibiki 2stars.png
Hibiki (Cheer Squad) bond  6 , level 1 2 star [1 1 1 0] 0 0 [0 0 0]
totaltime:26.838050842285156
student example\hibiki beeg no bord.png
Hibiki (Cheer Squad) bond  2 , level 1 1 star [11  1  0  0] 0 0 [0 0 0]
totaltime:24.102571487426758
student example\hibiki beeg.png
Hibiki (Cheer Squad) bond  2 , level 1 1 star [11  1  0  0] 0 0 [0 0 0]
totaltime:26.857065677642822
student example\hibiki ipad.png
Hibiki (Cheer Squad) bond  1 , level 1 1 star [1 1 0 0] 0 0 [0 0 0]
totaltime:66.34612250328064
student example\himari ipad.png
Himari bond  3 , level 75 3 star [5 4 4 7] 0 0 [0 0 0]
totaltime:66.97732353210449
student example\hinata ipad.png
Hinata bond  13 , level 75 3 star [3 1 4 4] 0 0 [6 1 4]
totaltime:72.93397688865662
student example\iori ipad.png
Iori bond  22 , level 80 5 star [ 5 10  4 10] 2 40 [7 6 7]
totaltime:78.23025918006897
student example\izumi ipad.png
Izumi bond  20 , level 10 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:80.82967138290405
student example\junko ipad.png
Junko bond  20 , level 71 4 star [4 4 4 7] 0 0 [5 2 3]
totaltime:79.8672103881836
student example\juri ipad.png
Juri bond  20 , level 7 4 star [1 1 1 1] 0 0 [0 0 0]
totaltime:79.65853452682495
student example\kirino ipad.png
Kirino bond  18 , level 38 3 star [1 1 1 1] 0 0 [5 5 5]
totaltime:78.06802272796631
student example\maki beeg.png
Maki bond  24 , level 80 5 star [ 5 10 10 10] 3 50 [7 6 6]
totaltime:33.057496309280396
student example\maki.png
Maki bond  24 , level 80 5 star [ 5 10 10 10] 3 50 [7 6 6]
totaltime:5.706263303756714
student example\marina.png
Marina bond  8 , level 1 3 star [11  1 11 11] 0 0 [0 0 0]
totaltime:29.832996606826782
student example\mashiro ipad.png
Mashiro bond  17 , level 6 5 star [3 1 1 7] 3 50 [0 0 0]
totaltime:80.93249106407166
student example\michiru ipad.png
Michiru bond  14 , level 15 4 star [1 1 1 1] 0 0 [1 0 0]
totaltime:81.10514974594116
student example\midori ipad.png
Midori bond  19 , level 75 5 star [5 4 8 9] 1 30 [7 6 6]
totaltime:73.19601821899414
student example\miku ipad.png
Hatsune Miku bond  20 , level 1 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:72.99165320396423
student example\mutsuki.png
Mutsuki bond  23 , level 80 5 star [ 5 10 10  7] 3 50 [7 6 6]
totaltime:29.909948110580444
student example\natsu ipad.png
Natsu bond  20 , level 75 3 star [4 7 6 7] 0 0 [7 7 6]
totaltime:80.68304109573364
student example\neru ipad.png
Neru bond  8 , level 76 3 star [1 1 1 1] 0 0 [5 6 6]
totaltime:78.8661880493164
student example\nodoka ipad.png
Nodoka bond  20 , level 45 4 star [5 7 1 7] 0 0 [2 1 2]
totaltime:68.61328935623169
student example\ochin ipad.png
Chinatsu (Hot Spring) bond  12 , level 70 3 star [ 5 10  1  1] 0 0 [6 3 1]
totaltime:69.00100708007812
student example\saya ipad.png
Saya bond  20 , level 35 4 star [1 1 1 1] 0 0 [1 1 1]
totaltime:68.90614700317383
student example\shimiko ipad.png
Shimiko bond  20 , level 8 5 star [1 1 1 1] 1 30 [0 0 0]
totaltime:68.79639077186584
student example\shizuko ipad.png
Shizuko bond  17 , level 75 5 star [4 1 1 1] 0 0 [7 6 5]
totaltime:68.79281854629517
student example\shun ipad.png
Shun (Small) bond  20 , level 2 3 star [1 1 1 1] 0 0 [0 0 0]
totaltime:71.81893944740295
student example\sizumi ipad.png
Izumi (Swimsuit) bond  20 , level 35 4 star [1 1 1 1] 0 0 [4 3 5]
totaltime:67.70295906066895
student example\sumire ipad.png
Sumire bond  20 , level 12 5 star [1 1 1 1] 1 30 [0 0 0]
totaltime:63.87843370437622
student example\suzumi ipad.png
Suzumi bond  18 , level 4 4 star [4 1 1 1] 0 0 [0 0 0]
totaltime:62.832072019577026
student example\tsurugi ipad.png
Tsurugi bond  15 , level 1 4 star [1 1 1 1] 0 0 [0 0 0]
totaltime:64.0025405883789
student example\ui ipad.png
Ui bond  17 , level 75 3 star [5 7 7 7] 0 0 [7 6 6]
totaltime:62.80451416969299
student example\yoshimi ipad.png
Yoshimi bond  20 , level 56 5 star [2 1 1 1] 1 30 [1 1 1]
totaltime:64.14345192909241
student example\yuuk ipad.png
Yuuka (Track) bond  4 , level 75 3 star [ 5 10  7  7] 0 0 [7 7 7]
totaltime:63.14710831642151
>>> 
