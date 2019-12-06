from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Create your views here.

@api_view(['POST'])
def index(req):
    data = req.data
    print(req.data)
    comments = data.get("comments")
    print(comments)
    from .tokenization.dict_models import LongMatchingTokenizer
    from .word_embedding.word2vec_gensim import Word2Vec
    from .text_classification.short_text_classifiers import BiDirectionalLSTMClassifier, load_synonym_dict
    keras_text_classifier = None
    word2vec_model = None
    # Please give the correct paths
    # Load word2vec model from file. If you want to train your own model, please go to README or check word2vec_gensim.py
    word2vec_model = Word2Vec.load(dir_path + '/models/pretrained_word2vec.bin')

    # Load tokenizer model for word segmentation. If you want to train you own model,
    # please go to README or check crf_tokenizer.py
    tokenizer = LongMatchingTokenizer()
    sym_dict = load_synonym_dict(dir_path  + '/data/sentiment/synonym.txt')
    keras_text_classifier = BiDirectionalLSTMClassifier(tokenizer=tokenizer, word2vec=word2vec_model.wv,
                                                        model_path=dir_path + '/models/sentiment_model.h5',
                                                        max_length=200, n_epochs=10,
                                                        sym_dict=sym_dict)
    # Load and prepare data
    # X, y = keras_text_classifier.load_data(['data/sentiment/samples/positive.txt',
    #                                     'data/sentiment/samples/negative.txt'],
    #                                     load_method=keras_text_classifier.load_data_from_file)

    # Train your classifier and test the model
    # keras_text_classifier.train(X, y)
    label_dict = {0: 'tích cực', 1: 'tiêu cực', 2: 'bình thường'}
    # test_sentences = ['Dở thế', 'Hay thế', 'phim chán thật', 'nhảm quá']
    test_sentences = comments
    labels = keras_text_classifier.classify(test_sentences, label_dict=label_dict)
    print(labels)  # Output: ['tiêu cực', 'tích cực', 'tiêu cực', 'tiêu cực']
    
    return Response(labels, status=status.HTTP_200_OK)
