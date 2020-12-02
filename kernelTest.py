""" 
Keep kernel implementations in here.
"""

import numpy as np
from collections import defaultdict, Counter
from functools import wraps
from tqdm import tqdm

from tweebo_parser import API, ServerError
from nltk.parse.dependencygraph import DependencyGraph
import graphviz
from nltk.corpus import treebank
import nltk


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
        X_prime = X if not X_prime else X_prime
        kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)

        #can be optimized
        for i in tqdm(range(len(X))):
            for j in range(len(X_prime)):
                kernel_matrix[i,j] = self.evaluate(X[i],X_prime[j])
        
        return kernel_matrix

class GraphKernel(Kernel):
    def __init__(self, *, ngram_length):
        self.ngram_length = ngram_length

    def add_root_node(self, list_conll_sentences):
        '''
        This adds the ROOT relation to CoNLL formatted data.
        '''
        temp_list_conll_sentences = []
        for conll_sentences in list_conll_sentences:
            temp_conll_sentences = []
            for sentence in conll_sentences.split('\n'):
                sentence = sentence.split('\t')
                if int(sentence[6]) == 0:
                    sentence[7] = 'ROOT'
                temp_conll_sentences.append('\t'.join(sentence))
            conll_sentences = '\n'.join(temp_conll_sentences)
            temp_list_conll_sentences.append(conll_sentences)
        return temp_list_conll_sentences
    
    def traverse(self, root, s, paths):
        for child in root:
            #check if its a leaf
            if type(child) == nltk.tree.Tree:
                self.traverse(child,s + " " +  child.label(), paths)
            else:
                paths.append(s+" " +str(child))
    
    def generate_ngrams(self, doc):
        NGrams = set()
        for i in range(len(doc)-self.ngram_length+1):
            NGrams.add(doc[i:i+self.ngram_length])
        return NGrams
    
    def evaluateNgram(self, s, t):
        X = self.generate_ngrams(s)
        Xprime = self.generate_ngrams(t)
        if len((X | Xprime)) == 0:
            return 1
        else:
            num = len(X & Xprime)
            den = len(X | Xprime)
            return num/den

    @cache_decorator()
    def evaluate(self, s, t):
        tweebo_api = API()
        text_data_s = [s]
        text_data_t = [t]
        max = 0
        try:
            #parse the raw string into two different lanugage representation formats
            #result_stanford = tweebo_api.parse_stanford(text_data)
            result_conll_s = tweebo_api.parse_conll(text_data_s)
            result_conll_t = tweebo_api.parse_conll(text_data_t)

            nltk_result_s = self.add_root_node(result_conll_s)
            nltk_result_t = self.add_root_node(result_conll_t)
            dep_tree_s = DependencyGraph(nltk_result_s[0]).tree()
            dep_tree_t = DependencyGraph(nltk_result_t[0]).tree()
            #dep_tree.draw()
            paths_s = []
            self.traverse(dep_tree_s,dep_tree_s.label(),paths_s)

            paths_t = []
            self.traverse(dep_tree_t,dep_tree_t.label(),paths_t)

            #compute N-gram on paths
            
            max = 0
            for string_s in paths_s:
                for string_t in paths_t:
                    result = self.evaluateNgram(string_s,string_t)
                    if result > max:
                        max = result
            return max
            
        except ServerError as e:
            print(f'{e}\n{e.message}')
        