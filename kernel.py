#outputs a kernel matrix 
#needs the dependency trees (in string format?)

from nltk.tree import Tree

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
        subtrees.append(Tree(parent, [graph]))
        for siblings in siblingTrees:
            #combine this permutation with each of the sibling permutations
            combined = [graph]
            combined.extend([sibling for sibling in siblings])
            subtrees.append(Tree(parent, combined))
    return subtrees

#Helper function to return a list of all subgraphs (trees) of a tree.
#Since recursive, second return of function is a list of only graphs
#that contain the parent node, to help in building higher subgraphs.
#This can be ignored when calling, first return is all possible subgraphs.
def getSubgraphs(tree):
    #sometimes leaves are represented as strings instead of trees
    if type(tree) is str:
        allGraphList = []
        topGraphList = []
        #turn the string into a tree
        parent = Tree(tree, [])
        #add to the lists and return
        allGraphList.append(parent)
        topGraphList.append(parent)
        return (allGraphList, topGraphList)
    #if the leaf is actually a tree, it will have no children
    elif len(tree) < 1:
        #return lists with the leaf
        allGraphList = []
        topGraphList = []
        allGraphList.append(tree)
        topGraphList.append(tree)
        return (allGraphList, topGraphList)
    else:
        #otherwise, if it does have children
        allGraphList = []
        topGraphList = []
        topChildGraphs = []
        #go through each of its children
        for i in range(len(tree)):
            #recursively find all of the subtrees of the children
            allChild, topChild = getSubgraphs(tree[i])
            #add these subgraphs to the allgraphs list
            allGraphList.extend(allChild)
            #keep track of the top subtrees of each child
            topChildGraphs.append(topChild)
        #permute together all of the possibile subtrees for each child with their siblings
        siblings = siblingMatch(tree.label(), topChildGraphs, 0)
        #also add the root parent
        parent = Tree(tree.label(), [])
        #add these to the lists and return
        topGraphList.append(parent)
        allGraphList.append(parent)
        topGraphList.extend(siblings)
        allGraphList.extend(siblings)
        return (allGraphList, topGraphList)

from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import numpy as np

# display a tree as a png
def printTree(tree):
    base64_string = tree3._repr_png_()
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    plt.imshow(np.array(image))
