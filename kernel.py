#outputs a kernel matrix 
#needs the dependency trees (in string format?)

from nltk.tree import Tree, ImmutableTree
from nltk.parse.dependencygraph import DependencyGraph

#hide warning for parsing graph without root element, we manually search for these roots
import warnings
warnings.filterwarnings("ignore", message="The graph doesn't contain a node that depends on the root element.")


#Convert a conll string representation of a tweet into a list of dependency trees
#One tree per root (generally one root per sentence)
def conllToTrees(conllString):
    dep = DependencyGraph(conllString)
    roots = {k: v for k, v in dep.nodes.items() if v.get('head') == 0}
    trees = []
    for root in roots:
        tree = dep._tree(root)
        trees.append(tree)
    return trees

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

from collections import defaultdict, Counter
from functools import wraps
from tqdm import tqdm


def cache_decorator():
    """
    Cache decorator. Stores elements to avoid repeated computations.
    For more details see: https://stackoverflow.com/questions/36684319/decorator-for-a-class-method-that-caches-return-value-after-first-access
    """
    def wrapper(function):
        """
        Return element if in cache. Otherwise compute and store.
        """
        cache = {}

        @wraps(function)
        def element(*args):
            if args in cache:
                result = cache[args]
            else:
                result = function(*args)
                cache[args] = result
            return result

        def clear():
            """
            Clear cache.
            """
            cache.clear()

        # Clear the cache
        element.clear = clear
        return element
    return wrapper


class Kernel(object):
    """ Abstract kernel object.
    """
    def evaluate(self, s, t):
        """
        Kernel function evaluation.
        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.
        Returns:
            A float from evaluating K(s,t)
        """
        raise NotImplementedError()

    def compute_kernel_matrix(self, *, X, X_prime=None):
        """
        Compute kernel matrix. Index into kernel matrix to evaluate kernel function.
        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.
        Returns:
            A compressed sparse row matrix of floats with each element representing
            one kernel function evaluation.
        """

        #for training, optimize by only computing upper triangular
        if not X_prime:
            #replace X_prime with training data
            X_prime = X  
            kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)
            for i in tqdm(range(len(X))):
                for j in range(len(X_prime)):
                    #compute upper triangular
                    if (j > i):
                        kernel_matrix[i,j] = self.evaluate(X[i],X_prime[j])
                    #diagonal will contain only 1's
                    elif (j == i):
                        kernel_matrix[i,j] = 1
                    #copy from upper triangular
                    else:
                        kernel_matrix[i,j] = kernel_matrix[j,i]
        #testing                
        else:
            kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)
            for i in tqdm(range(len(X))):
                for j in range(len(X_prime)):
                    kernel_matrix[i,j] = self.evaluate(X[i],X_prime[j])
        return kernel_matrix


class SimplePairsKernel(Kernel):

    @cache_decorator()
    def evaluate(self, tweet1, tweet2):
        """
        Args:
            tweet1: a tweet in (string, conll) tuple form.
            tweet2: a tweet in (string, conll) tuple form.
        Returns:
            A float from evaluating K(tweet1,tweet2)
        """
        trees1 = conllToTrees(tweet1[1])
        vocabulary1 = set()
        for tree in trees1:
            vocabulary1 = vocabulary1.union(getSyntacticPairs(tree))

        trees2 = conllToTrees(tweet2[1])
        vocabulary2 = set()
        for tree in trees2:
            vocabulary2 = vocabulary2.union(getSyntacticPairs(tree))

        common = len(vocabulary1.intersection(vocabulary2))
        total = len(vocabulary1.union(vocabulary2))
        
        if total == 0:
            return 1
        else:
            return common / total

class SimpleSubgraphsKernel(Kernel):

    @cache_decorator()
    def evaluate(self, tweet1, tweet2):
        """
        Args:
            tweet1: a tweet in (string, conll) tuple form.
            tweet2: a tweet in (string, conll) tuple form.
        Returns:
            A float from evaluating K(tweet1,tweet2)
        """
        trees1 = conllToTrees(tweet1[1])
        vocabulary1 = set()
        for tree in trees1:
            vocabulary1 = vocabulary1.union(getSubgraphs(tree)[0])

        trees2 = conllToTrees(tweet2[1])
        vocabulary2 = set()
        for tree in trees2:
            vocabulary2 = vocabulary2.union(getSubgraphs(tree)[0])

        common = len(vocabulary1.intersection(vocabulary2))
        total = len(vocabulary1.union(vocabulary2))
        
        if total == 0:
            return 1
        else:
            return common / total

