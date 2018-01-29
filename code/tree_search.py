import TreeNode from tree_node.py

def predict(model, timestep_data, depth=3, variance=0.05, num_variance=2, holdings=[10000, 10000]):
  i = 0
  # tree search
  # first, establish the root node and add its children
""" traverseVertex() recursive loop:
      backpropReturns() # propogate returns and numvisits to parent nodes
      checkMaxDepth() -> return if true # stop going deeper if we hit depth limit
      updateTimeSeries() # update time series with the new price
      predict() # use neural net to predict
      generateChildren() # w/price variance & new holdings/returns from action for each child
""" 
  node = TreeNode(price=timestep_data[len(timestep_data) - 1], holdings=holdings, root_price=timestep_data[len(timestep_data) - 1])
  # generate initial children
  generateChildren(node, variance, num_variance, model.predict(timestep_data)[1][0])
  for key, item in node.children:
    traverseVertex(item, depth, 0, timestep_data)
  
  # when we're done updating all the returns w/tree search, return the final best choice
  # highest avg returns wins
  j = 0
  maxIndex = 0
  while j < 3:
    if node.children[j].avg_returns > node.children[maxIndex].avg_returns:
      maxIndex = j
    j += 1
  return maxIndex # 0=buy, 1=sell, 2=hold

def traverseVertex(node, max_depth=1, depth, timeseries):
  backpropReturns(node)
  if depth == max_depth:
    return
  else:
    # Update time series w/new price
    series = list(timeseries)
    series.remove(0)
    series.append([node.price])



def generateChildren(node, variance, num_variance, price_change, firstGen):
  actions = ['buy', 'sell', 'hold']
  for key, item in actions:
    i = 0
    initial_variance = -1 * variance * num_variance
    # if it's the first depth, we technically start at depth 0, so don't do variance
    if firstGen:
      initial_variance = 0
      num_variance = 0
    while i < num_variance * 2 + 1:
      # add a node with this variance and action
      initial_variance = initial_variance + variance
      # calculate the new price
      new_price = node.price * price_change * (1 + initial_variance)
      newNode = TreeNode(price=new_price, parent=node, holdings=list(node.holdings), root_price=node.root_price)

      if item == 'buy':
        # buy 1000$ or max cash, whichever is less, worth
        amt = newNode.holdings[1]
        if amt >= 1000:
          amt = 1000
        newNode.holdings[0] = (amt / node.price) + newNode.holdings[0]
        newNode.holdings[1] = newNode.holdings[1] - amt
      if item == 'sell':
        # sell 1000$ or max cash, whichever is less, worth
        amt = newNode.holdings[0] * node.price
        if amt >= 1000:
          amt = 1000
        newNode.holdings[0] = newNode.holdings[0] - (amt / node.price)
        newNode.holdings[1] = newNode.holdings[1] + amt
      node.children.append(newNode)
      i += 1


def backpropReturns(node):
  rootNode = node.parent
  currentNode = node
  while rootNode != None:
    # Weighted average of returns update
    rootNode.avg_returns = rootNode.avg_returns * rootNode.num_visits + ((currentNode.holdings[0] * currentNode.price + currentNode.holdings[1]))
    rootNode.num_visits = rootNode.num_visits + 1
    rootNode.avg_returns = rootNode.avg_returns / rootNode.num_visits

    # traverse up the tree
    rootNode = rootNode.parent
    currentNode = currentNode.parent