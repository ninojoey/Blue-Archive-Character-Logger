import os

directory = os.curdir
scale = 1.0
for fileName in os.listdir(directory):
    originalPath = os.path.join(directory, fileName)
    string = "skill level level"
    if string in fileName:
        replaceString = fileName.replace("skill level level", "skill level")
        f = os.path.join(directory, replaceString)
        os.rename(originalPath, f)
