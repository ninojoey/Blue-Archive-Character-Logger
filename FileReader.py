import json
from PlayerData import PlayerData





importedTextFile = open("example.txt", "r")
##importedData = input("Please copy and paste your exported data from the website: ")


# reads all the lines of the input and stores each line into an index of an array
content = importedTextFile.readlines()


# the import should only be one line, therefore the length of the array should be 1
if len(content) != 1:
    print("text file bad structure")


# what json.loads(str) does is that it turns a json-structured string into a python dictionary
importedPlayerData = json.loads(content[0])

# create an instance of PlayerData
playerData = PlayerData()


if "exportVersion" in importedPlayerData:
    playerData.exportVersion = importedPlayerData.pop("exportVersion")
    
if "characters" in importedPlayerData:
    playerData.characters = importedPlayerData.pop("characters")
    
if "disabled_characters" in importedPlayerData:
    playerData.disabled_characters = importedPlayerData.pop("disabled_characters")
    
if "owned_materials" in importedPlayerData:
    playerData.owned_materials = importedPlayerData.pop("owned_materials")
    
if "site_version" in importedPlayerData:
    playerData.site_version = importedPlayerData.pop("site_version")
    
if "character_order" in importedPlayerData:
    playerData.character_order = importedPlayerData.pop("character_order")
    
if "page_theme" in importedPlayerData:
    playerData.page_theme = importedPlayerData.pop("page_theme")
    
if "groups" in importedPlayerData:
    playerData.groups = importedPlayerData.pop("groups")
    
if "server" in importedPlayerData:
    playerData.server = importedPlayerData.pop("server")

if "language" in importedPlayerData:
    playerData.language = importedPlayerData.pop("language")
    
if "level_cap" in importedPlayerData:
    playerData.level_cap = importedPlayerData.pop("level_cap")
    
if "events_data" in importedPlayerData:
    playerData.events_data = importedPlayerData.pop("events_data")


if len(importedPlayerData) > 0:
    print("YOU MISSED SOMETHING")
else:
    print("Successfully extracted data")

print(playerData)

