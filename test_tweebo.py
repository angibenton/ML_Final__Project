#BEFORE RUNNING THIS CODE: 
#Install Docker https://docs.docker.com/get-docker/
#docker run -p 8000:8000 -d --rm mooreap/tweeboparserdocker

from tweebo_parser import API, ServerError
from nltk.parse.dependencygraph import DependencyGraph
import graphviz


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
text_data = ['Guangdong University of Foreign Studies is located in Guangzhou.',
             'Lucy is in the sky with diamonds.']
try:
    #parse the raw string into two different lanugage representation formats
    result_stanford = tweebo_api.parse_stanford(text_data)
    result_conll = tweebo_api.parse_conll(text_data)

    nltk_result = add_root_node(result_conll)
    nltk_dep_tree = DependencyGraph(nltk_result[0])
    
    #print(result_stanford)
    #print(result_conll)
    #print(nltk_dep_tree)
    nltk_dep_tree.tree() #this doesnt output the image like in the example?
except ServerError as e:
    print(f'{e}\n{e.message}')