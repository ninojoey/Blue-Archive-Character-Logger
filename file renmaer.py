import os

directory = "stats level"
scale = 1.0
for fileName in os.listdir(directory):
    originalPath = os.path.join(directory, fileName)
    replaceString = fileName.replace("level", "stats level")
    
    f = os.path.join(directory, replaceString)

    os.rename(originalPath, f)
