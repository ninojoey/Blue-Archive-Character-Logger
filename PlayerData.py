class PlayerData():
    def __init__(self, exportVersion = "2", characters = [], disabled_characters = [], owned_materials = [], site_version = "1.3.5", character_order = [], page_theme = "dark", groups = [], server = "Global", language = "En", level_cap = 83, events_data = []):
        self.exportVersion = exportVersion
        self.characters = characters
        self.disabled_characters = disabled_characters
        self.owned_materials = owned_materials
        self.site_version = site_version
        self.character_order = character_order
        self.page_theme = page_theme
        self.groups = groups
        self.server = server
        self.language = language
        self.level_cap = level_cap
        self.events_data = events_data

    def __str__(self):
        returnString = ""
        return returnString
