""" 
Keep kernel implementations in here.
"""

import numpy as np
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
        X_prime = X if not X_prime else X_prime
        kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)

        #can be optimized
        for i in tqdm(range(len(X))):
            for j in range(len(X_prime)):
                kernel_matrix[i,j] = self.evaluate(X[i],X_prime[j])
        
        return kernel_matrix


class NgramKernel(Kernel):
    def __init__(self, *, ngram_length):
        """
        Args:
            ngram_length: length to use for n-grams
        """
        self.ngram_length = ngram_length


    def generate_ngrams(self, doc):
        """
        Generate the n-grams for a document.

        Args:
            doc: A string corresponding to a document.

        Returns:
            Set of all distinct n-grams within the document.
        """
        # TODO: Implement this!
        NGrams = set()
        for i in range(len(doc)-self.ngram_length+1):
            NGrams.add(doc[i:i+self.ngram_length])
        return NGrams


    @cache_decorator()
    def evaluate(self, s, t):
        """
        n-gram kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        # TODO: Implement this!
        X = self.generate_ngrams(s)
        Xprime = self.generate_ngrams(t)
        if len((X | Xprime)) == 0:
            return 1
        else:
            num = len(X & Xprime)
            den = len(X | Xprime)
            return num/den

class TFIDFKernel(Kernel):
    def __init__(self, *, X, X_prime=None):
        """
        Pre-compute tf-idf values for each (document, word) pair in dataset.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Sets:
            tfidf: You will use this in the evaluate function.
        """
        self.tfidf = self.compute_tfidf(X, X_prime)
        

    def compute_tf(self, doc):
        """
        Compute the tf for each word in a particular document.
        You may choose to use or not use this helper function.

        Args:
            doc: A string corresponding to a document.

        Returns:
            A data structure containing tf values.
        """
        doc = doc.split()
        tfDict = dict.fromkeys(doc,0)
        for word in doc:
            tfDict[word] += 1

        for word, count in tfDict.items():
            tfDict[word] = count/float(len(doc))

        return tfDict


    def compute_df(self, X, vocab):
        """
        Compute the df for each word in the vocab.
        You may choose to use or not use this helper function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            vocab: A set of distinct words that occur in the corpus.

        Returns:
            A data structure containing df values.
        """
        # TODO: Implement this!
        dfDict = dict.fromkeys(vocab,0)
        for word in vocab:
            for doc in X:
                if word in doc.split():
                    dfDict[word] += 1
        
        return dfDict


    def compute_tfidf(self, X, X_prime):
        """
        Compute the tf-idf for each (document, word) pair in dataset.
        You will call the helper functions to compute term-frequency 
        and document-frequency here.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A data structure containing tf-idf values. You can represent this however you like.
            If you're having trouble, you may want to consider a dictionary keyed by 
            the tuple (document, word).
        """
        # Concatenate collections of documents during testing
        if X_prime:
            X = X + X_prime
        #TODO: Implement this!
        vocab = set()
        for s in X:
            vocab.update(s.split())
        dfDict = self.compute_df(X,vocab)

        tfidfDict = {}
        for document in X:
            tfDict = self.compute_tf(document)
            for word in document.split():
                tfidfDict[(document,word)] = tfDict[word] * np.log(len(X)/ (dfDict[word] + 1))

        return tfidfDict

    @cache_decorator()
    def evaluate(self, s, t):
        """
        tf-idf kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        #TODO: Implement this!
        #tfx = self.compute_tf(s)
        tfxprime = self.compute_tf(t)
        k = 0
        for word in s.split(): #for each distinct word in x
            # if word in t.split() and (s,word) in self.tfidf.keys():
            if word in t.split():
                #print(tfxprime)
                #print(word)
                #print(t)
                freq = tfxprime[word]
                k = k + freq * self.tfidf[(s,word)]

        return k

        

#     def compute_tf(self, doc):
#         doc = doc.split()
#         wordDic = dict.fromkeys(doc,0)
#         for word in doc:
#             wordDic[word] += 1

#         tfDict = {}
#         for word, count in wordDic.items():
#             tfDict[word] = count/float(len(doc))
        
#         return tfDict