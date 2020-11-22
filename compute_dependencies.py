#--------- all tweebo-dependent code should be in this file -----------
#BEFORE RUNNING THIS CODE: 
#Install Docker https://docs.docker.com/get-docker/
#docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker
from tweebo_parser import API, ServerError
from nltk.parse.dependencygraph import DependencyGraph
import graphviz
from nltk.corpus import treebank

def add_root_node(list_conll_sentences):
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

tweebo_api = API() # Assumes server is running locally at 0.0.0.0:8000

#input: the preprocessed datasets with columns: tweet (string) and class (-1 or 1)
#output: .csv with columns: tweet (string), dependency tree (conll string representation), class (-1 or 1)
