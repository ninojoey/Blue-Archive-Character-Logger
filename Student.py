class Student:
    def __init__(self):
        self.ID = "99999"
        self.name = "Dummy"
        self.currentStats = Stats()
        self.targetStats = Stats()
        self.enabled = False
        self.eleph = Eleph()

    def __init__(self, ID, name, current, target, enabled, eleph):
        self.ID = ID
        self.name = name
        self.current = current
        self.target = target
        self.enabled = enabled
        self.eleph = eleph
