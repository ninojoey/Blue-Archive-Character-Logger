import json
import requests

# maybe turn this into an updater? and export the update 
# to like a txt file so everyone else can work with it



CHAR_JSON_URL = "https://raw.githubusercontent.com/JustinL163/BA-Resource-Planner/main/planner/json/charlist.json"
ITEM_JSON_URL = "https://raw.githubusercontent.com/JustinL163/BA-Resource-Planner/main/planner/json/localisations.json"



class bidict:
    def __init__(self):
        self.dict = {}
        self.revDict = {}
        
    def __init__(self, dictionary):
        self.dict = dictionary
        self.revDict = {}
        for key, value in dictionary.items():
            self.revDict[value] = key

    def add(self, key, value):
        self.dict[key] = value
        self.revDict[value] = key

    def get(self, key):
        if key in self.dict:
            return self.dict[key]
        elif key in self.revDict:
            return self.revDict[key]
        else:
            return "Key does not exist"

    def toString(self):
        bidictToString = "{"
        for key in self.dict:
            bidictToString += "{}:{}, ".format(key, self.dict[key])

        return bidictToString
        





# get the dictionary of studentID:studentInfo
resp = requests.get(CHAR_JSON_URL)
studentsDict =  json.loads(resp.text)


studentBidict = bidict({})
studentIDList = []
studentNameList = []

for studentID in studentsDict:
    studentInfo = studentsDict[studentID]
    if "Name" in studentInfo:
        studentName = studentInfo["Name"]
        
        studentBidict.add(studentID, studentName)
        studentIDList.append(studentID)
        studentNameList.append(studentName)






# this json is  first split into languages. we want the english one
# then it's split into items and characters. we want only the items

resp = requests.get(ITEM_JSON_URL)
langLocals =  json.loads(resp.text)
categories = langLocals["En"]
itemDict = categories["Items"]

itemBidict = bidict({})
itemIDList = []
itemNameList = []

for itemID in itemDict:
    itemInfo = itemDict[itemID]
    if "Name" in itemInfo:
        itemName = itemInfo["Name"]
        
        itemBidict.add(itemID, itemName)
        itemIDList.append(itemID)
        itemNameList.append(itemName)
    else:
        if itemID in studentIDList:
            studentIndex = studentIDList.index(itemID)
            itemName = "{}'s Eleph".format(studentNameList[studentIndex])

            itemBidict.add(itemID, itemName)
            itemIDList.append(itemID)
            itemNameList.append(itemName)
                                
        
print(itemBidict.toString())
