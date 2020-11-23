#outputs a kernel matrix 
#needs the dependency trees (in string format?)

def getSubgraphs(tree):
    if type(tree) is str:
        allGraphList = []
        topGraphList = []
        node = Tree(tree, [])
        allGraphList.append(node)
        topGraphList.append(node)
        return (allGraphList, topGraphList)
    elif len(tree) < 1:
        allGraphList = []
        topGraphList = []
        allGraphList.append(tree)
        topGraphList.append(tree)
        return (allGraphList, topGraphList)
    else:
        allGraphList = []
        topGraphList = []
        allLeftGraphs, topLeftGraphs = getSubgraphs(tree[0])
        allRightGraphs = []
        topRightGraphs = []
        if (len(tree) > 1):
            allRightGraphs, topRightGraphs = getSubgraphs(tree[1])
        for rightGraph in topRightGraphs:
            for leftGraph in topLeftGraphs:
                subgraph = Tree(tree.label(), [leftGraph, rightGraph])
                topGraphList.append(subgraph)
                allGraphList.append(subgraph)
        for rightGraph in topRightGraphs:
            subgraph = Tree(tree.label(), [rightGraph])
            topGraphList.append(subgraph)
            allGraphList.append(subgraph)
        for leftGraph in topLeftGraphs:
            subgraph = Tree(tree.label(), [leftGraph])
            topGraphList.append(subgraph)
            allGraphList.append(subgraph)
        node = node = Tree(tree.label(), [])
        topGraphList.append(node)
        allGraphList.append(node)
        allGraphList.extend(allLeftGraphs)
        allGraphList.extend(allRightGraphs)
        return (allGraphList, topGraphList)
