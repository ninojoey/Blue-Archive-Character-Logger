from Stats import *
from Eleph import *


class Student:
    def __init__(self, name = "Dummy", bond = 0, level = 0, star = 0, skills = [0, 0, 0, 0], ueStar = 0, ueLevel = 0, gearTiers = [0, 0, 0], eleph = Eleph(), enabled = True):
        self.id = "99999"
        self.name = name
        self.current = Stats(bond, level, star, skills, ueStar, ueLevel, gearTiers)
        self.target = Stats(bond, level, star, skills, ueStar, ueLevel, gearTiers)
        self.eleph = eleph
        self.enabled = enabled

    def __str__(self):
        return str(vars(self))


##stud = Student("danny", 1, 2, 3, [4, 5, 6, 13], 7, 8, [9,10,11,12])
##
##print(stud)

##"characters":[{"id":"23000","name":"Airi","current":{"level":1,"bond":1,"star":2,"ue":0,"ue_level":0,"ex":1,"basic":1,"passive":0,"sub":0,"gear1":0,
##"gear2":0,"gear3":0},"target":{"level":"1","bond":"1","star":1,"ue":0,"ue_level":"0","ex":"1","basic":"1","passive":"0","sub":"0","gear1":"0","gear2":"0",
##"gear3":0},"eleph":{"owned":0,"unlocked":true,"cost":1,"purchasable":20,"farm_nodes":0,"node_refresh":false,"use_eligma":false,"use_shop":false},"enabled":true},...]
