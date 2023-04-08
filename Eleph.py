class Eleph:
    def __init__(self):
        self.owned = 0
        self.unlocked = False
        self.cost = 1
        self.purchasable = 20
        self.farm_nodes = 0
        self.node_refresh = False
        self.use_eligma = False
        self.use_shop = False
        
    def __init__(self, owned, unlocked, cost, purchasable, farm_nodes, node_refresh, use_eligma, use_shop):
        self.owned = owned
        self.unlocked = unlocked
        self.cost = cost
        self.purchasable = purchasable
        self.farm_nodes = farm_nodes
        self.node_refresh = node_refresh
        self.use_eligma = use_eligma
        self.use_shop = use_shop
