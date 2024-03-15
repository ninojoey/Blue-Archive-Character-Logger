class Stats:
    def __init__(self, bond = 0, level = 0, star = 0, skills = [0, 0, 0, 0], ueStar = 0, ueLevel = 0, gearTiers = [0, 0, 0]):
        self.bond = bond
        self.level = level
        self.star = star
        self.ex = skills[0]
        self.basic = skills[1]
        self.passive = skills[2]
        self.sub = skills[3]
        self.ue = ueStar
        self.ue_level = ueLevel
        self.gear1 = gearTiers[0]
        self.gear2 = gearTiers[1]
        self.gear3 = gearTiers[2]
        
    def __repr__(self):
        return str(vars(self))
        
    def __str__(self):
        return str(vars(self))


##stats = Stats()
##print(stats)
##current":{"level":1,"bond":1,"star":2,"ue":0,"ue_level":0,"ex":1,"basic":1,"passive":0,"sub":0,"gear1":0,"gear2":0,"gear3":0}

