#outputs a kernel matrix 
#needs the dependency trees (in string format?)

from nltk.tree import Tree, ImmutableTree

#A helper function to create all possible subtrees by matching all possible subtrees of the siblings to one another
def siblingMatch(parent, treeList, index):
    subtrees = []
    siblingTrees = []
    #if there is a next sibling
    if (index + 1 < len(treeList)):
            #get all of the permutations of the siblings to the right
            siblingTrees = siblingMatch(parent, treeList, index + 1)
            subtrees.extend(siblingTrees)
    #go through all the subtrees of this child
    for graph in treeList[index]:
        #add this permutation with no siblings
        subtrees.append(ImmutableTree(parent, [graph]))
        for siblings in siblingTrees:
            #combine this permutation with each of the sibling permutations
            combined = [graph]
            combined.extend([sibling for sibling in siblings])
            subtrees.append(ImmutableTree(parent, combined))
    return subtrees

#Helper function to return a list of all subgraphs (trees) of a tree.
#Since recursive, second return of function is a list of only graphs
#that contain the parent node, to help in building higher subgraphs.
#This can be ignored when calling, first return is all possible subgraphs.
def getSubgraphs(tree):
    #sometimes leaves are represented as strings instead of trees
    if type(tree) is str:
        allGraphList = set()
        topGraphList = []
        #turn the string into a tree
        parent = ImmutableTree(tree, [])
        #add to the lists and return
        allGraphList.add(parent)
        topGraphList.append(parent)
        return (allGraphList, topGraphList)
    #if the leaf is actually a tree, it will have no children
    elif len(tree) < 1:
        #return lists with the leaf
        allGraphList = set()
        topGraphList = []
        allGraphList.add(ImmutableTree(tree, []))
        topGraphList.append(ImmutableTree(tree, []))
        return (allGraphList, topGraphList)
    else:
        #otherwise, if it does have children
        allGraphList = set()
        topGraphList = []
        topChildGraphs = []
        #go through each of its children
        for i in range(len(tree)):
            #recursively find all of the subtrees of the children
            allChild, topChild = getSubgraphs(tree[i])
            #add these subgraphs to the allgraphs list
            allGraphList.update(allChild)
            #keep track of the top subtrees of each child
            topChildGraphs.append(topChild)
        #permute together all of the possibile subtrees for each child with their siblings
        siblings = siblingMatch(tree.label(), topChildGraphs, 0)
        #also add the root parent
        parent = ImmutableTree(tree.label(), [])
        #add these to the lists and return
        topGraphList.append(parent)
        allGraphList.add(parent)
        topGraphList.extend(siblings)
        allGraphList.update(siblings)
        return (allGraphList, topGraphList)

#helper function to return set of all pairs from dependency tree
def getSyntacticPairs(tree):
    pairs = set()
    #no pairs left if on leaves
    if type(tree) is str:
        return []
    elif len(tree) < 1:
        return []
    else:
        #add the pair between parent and each child
        for child in tree:
            #if child is leaf, might not be a tree, handle carefully
            childName = ""
            if type(child) is str:
                childName = child
            else:
                childName = child.label()
            #add the pair with that child
            pairs.add((tree.label(), childName))
            #add all of that child's pairs
            pairs.update(getSyntacticPairs(child))
    return pairs

from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import numpy as np

# display a tree as a png
def printTree(tree):
    base64_string = tree._repr_png_()
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    plt.figure()
    plt.imshow(np.array(image))
