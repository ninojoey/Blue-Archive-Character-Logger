class Eleph:
    def __init__(self, owned = 0, unlocked = True, cost = 1, purchasable = 20, farmNodes = 0, nodeRefresh = False, useEligma = False, useShop = False):
        self.owned = owned
        self.unlocked = unlocked
        self.cost = cost
        self.purchasable = purchasable
        self.farm_nodes = farmNodes
        self.node_refresh = nodeRefresh
        self.use_eligma = useEligma
        self.use_shop = useShop
    
    def __repr__(self):
        return str(vars(self))
    
    def __str__(self):
        return str(vars(self))

##"eleph":{"owned":0,"unlocked":true,"cost":1,"purchasable":20,"farm_nodes":0,"node_refresh":false,"use_eligma":false,"use_shop":false}
