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
text_data = ['!!!!!!""@__BrighterDays: I can not just sit up and HATE on another bitch .. I got too much shit going on!""',
             'I can not just sit up and HATE on another bitch .. I got too much shit going on!']
try:
    #parse the raw string into two different lanugage representation formats
    result_stanford = tweebo_api.parse_stanford(text_data)
    result_conll = tweebo_api.parse_conll(text_data)

    nltk_result = add_root_node(result_conll)
    nltk_dep_tree_0 = DependencyGraph(nltk_result[0])
    nltk_dep_tree_1 = DependencyGraph(nltk_result[1])
    
    #print(result_stanford)
    #print(result_conll)
    #print(nltk_result)
    #print(nltk_dep_tree.contains_cycle())
    tree_0 = nltk_dep_tree_0.tree()
    tree_1 = nltk_dep_tree_1.tree()
    #nltk_dep_tree.tree().view() 
    print(tree_0)
    for subtree in tree_0.subtrees():
        print(subtree)

    print(tree_1)
    for subtree in tree_1.subtrees():
        print(subtree)

    #TODO test a multi-sentence string!!
except ServerError as e:
    print(f'{e}\n{e.message}')