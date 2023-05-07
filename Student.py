class Student:
    def __init__(self, ID = "99999", name = "Dummy", current = Stats(), target = Stats(), enabled = False, eleph = Eleph()):
        self.ID = ID
        self.name = name
        self.current = current
        self.target = target
        self.enabled = enabled
        self.eleph = eleph
