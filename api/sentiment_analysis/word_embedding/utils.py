import re, os, string
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

# list stopwords
filename = dir_path + '/stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']
# print(list_stopwords)


def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def sentence_segment(text):
    sents = re.split("([.?!])?[\n]+|[.?!] ", text)
    return sents


def word_segment(sent):
    from tokenization.dict_models import LongMatchingTokenizer
    tokenizer = LongMatchingTokenizer() 
    sent = tokenizer.tokenize(sent)
    return ' '.join(sent)


path_to_corpus = 'data/word_embedding/AA'


# f_w = open('data/word_embedding/training/datatrain.txt', 'w')
# for i, sub_dir in enumerate(os.listdir(path_to_corpus)):
#     path_to_subdir = path_to_corpus + '/' + sub_dir
#     print(i, sub_dir, path_to_subdir)
#     for j, file_name in enumerate(os.listdir(path_to_subdir)):
#         print(file_name)
#         f_w = open('data/word_embedding/training/' + file_name, 'w')
#         with open(path_to_subdir + '/' + file_name) as f_r:
#             contents = f_r.read().strip().split('</doc>')
#             # print(contents[0])
#             for content in contents:
#                 if (len(content) < 5):
#                     continue
#                 content = clean_text(content)
#                 sents = sentence_segment(content)
#                 # print(sents)
#                 for sent in sents:
#                     if(sent != None):
#                         sent = word_segment(sent)
#                         sent = remove_stopword(normalize_text(sent))
#                         # print(sent.split())
#                         if(len(sent.split()) > 1):
#                             f_w.write(sent + '\n')
#             print ("Done ", i + 1, ':', j + 1)

for i, file_name in enumerate(os.listdir(path_to_corpus)):
    path_to_subdir = path_to_corpus + '/' + file_name
    print(i, file_name, path_to_subdir)

    f_w = open('data/word_embedding/training/' + file_name, 'w')
    with open(path_to_subdir) as f_r:
        contents = f_r.read().strip().split('</doc>')
        # print(contents[0])
        for content in contents:
            if (len(content) < 5):
                continue
            content = clean_text(content)
            sents = sentence_segment(content)
            print(sents)
            for sent in sents:
                if(sent != None):
                    sent = word_segment(sent)
                    sent = remove_stopword(normalize_text(sent))
                    # print(sent.split())
                    if(len(sent.split()) > 1):
                        f_w.write(sent + '\n')
        # print ("Done ", i + 1, ':', j + 1)

f_w.close()