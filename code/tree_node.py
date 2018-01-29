class TreeNode:
    def __init__(self, price, parent=None, children=[], avg_returns=1, num_visits=0, holdings=[10000, 10000], root_price=0):
        self.price = price
        self.parent = parent
        self.children = children
        self.avg_returns = avg_returns
        self.num_visits = num_visits
        self.holdings = holdings # 0 index = stock amt, 1 index=$