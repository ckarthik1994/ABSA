#!/usr/bin/env python

'''
**Aspect Category Detection for the 5th task of SemEval 2016**
Unconstrained Submission for the Restaurants domain

Run from the terminal:
>>> python acd_unconstrained_restaurants.py --train train.xml --test test.xml
'''

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os, warnings
    import numpy as np
    from collections import Counter
    import operator
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.svm import SVC
    import nltk
    from nltk.stem import PorterStemmer
    from xml.sax.saxutils import escape
    from sklearn.externals import joblib
    import acd_restaurants_train
except Exception, ex:
    sys.exit(str(ex))

warnings.filterwarnings("ignore") #to ignore sklearns deprecation warnings

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

class AspectCategoryClassifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        cats = joblib.load('./models/acd/rest/categories.pkl')
        self.cat_dict = Counter(cats)
        self.w2v_model = joblib.load('./models/acd/rest/w2v_model.pkl')

        self.unigrams_lexica = joblib.load('./models/acd/rest/unigrams_lexica.pkl')
        self.bigrams_lexica = joblib.load('./models/acd/rest/bigrams_lexica.pkl')
        self.bipos_lexica = joblib.load('./models/acd/rest/bipos_lexica.pkl')
        self.stemmed_unigrams_lexica = joblib.load('./models/acd/rest/stemmed_unigrams_lexica.pkl')
        self.stemmed_bigrams_lexica = joblib.load('./models/acd/rest/stemmed_bigrams_lexica.pkl')
        self.idf_dict = joblib.load('./models/acd/rest/idf_dict.pkl')
        self.category_centroids = joblib.load('./models/acd/rest/category_centroids.pkl')

        self.food_clf1 = joblib.load('models/acd/rest/food_clf1.pkl')
        self.drinks_clf1 = joblib.load('models/acd/rest/drinks_clf1.pkl')
        self.service_clf1 = joblib.load('models/acd/rest/service_clf1.pkl')
        self.ambience_clf1 = joblib.load('models/acd/rest/ambience_clf1.pkl')
        self.location_clf1 = joblib.load('models/acd/rest/location_clf1.pkl')
        self.restaurant_clf1 = joblib.load('models/acd/rest/restaurant_clf1.pkl')
        self.general_clf1 = joblib.load('models/acd/rest/general_clf1.pkl')
        self.price_clf1 = joblib.load('models/acd/rest/price_clf1.pkl')
        self.quality_clf1 = joblib.load('models/acd/rest/quality_clf1.pkl')
        self.style_clf1 = joblib.load('models/acd/rest/style_clf1.pkl')
        self.misc_clf1 = joblib.load('models/acd/rest/misc_clf1.pkl')

        self.food_clf2 = joblib.load('models/acd/rest/food_clf2.pkl')
        self.drinks_clf2 = joblib.load('models/acd/rest/drinks_clf2.pkl')
        self.service_clf2 = joblib.load('models/acd/rest/service_clf2.pkl')
        self.ambience_clf2 = joblib.load('models/acd/rest/ambience_clf2.pkl')
        self.location_clf2 = joblib.load('models/acd/rest/location_clf2.pkl')
        self.restaurant_clf2 = joblib.load('models/acd/rest/restaurant_clf2.pkl')
        self.general_clf2 = joblib.load('models/acd/rest/general_clf2.pkl')
        self.price_clf2 = joblib.load('models/acd/rest/price_clf2.pkl')
        self.quality_clf2 = joblib.load('models/acd/rest/quality_clf2.pkl')
        self.style_clf2 = joblib.load('models/acd/rest/style_clf2.pkl')
        self.misc_clf2 = joblib.load('models/acd/rest/misc_clf2.pkl')


    def identifyQueryCategory(self, query):
        print('Done!')
        print('Creating test feature vectors...')
        
        test_sentences1 = []
        test_sentences2 = []
        words = (re.findall(r"[\w']+", query.lower()))
        
        sentence_without_stopwords = ""
        for w in words:
            if w not in stopwords:
                sentence_without_stopwords = sentence_without_stopwords + " " + w
        #clean the words, so we can get their embeddings
        clean_words = acd_restaurants_train.clean(sentence_without_stopwords).split()

        #calculate the embedding for the words of the current sentence
        sentence_vector_feats = []
        words_with_embeds = []
        for w in set(clean_words):
            word_vector_feats = []
            if w in self.w2v_model:
                words_with_embeds.append(w)
                for vector in self.w2v_model[w]:
                    word_vector_feats.append(vector)
                sentence_vector_feats.append(word_vector_feats)

        #calculate the centroid of the embeddings of the sentence (using idf)
        centroid_feats = []
        for vec_num in range(0,200):
            sum_vectors = 0.
            sum_idf = 0.
            for w_index, word_vector in enumerate(sentence_vector_feats):
                sum_vectors = sum_vectors + (word_vector[vec_num] * self.idf_dict[words_with_embeds[w_index]])
                sum_idf = sum_idf + self.idf_dict[words_with_embeds[w_index]]
            centroid = sum_vectors / (sum_idf) if sum_idf > 0. else 0.
            centroid_feats.append(centroid)

        normalized_centroid_feats = acd_restaurants_train.normalize_horizontal(centroid_feats)

        #compute the cosine similarity of the centroid of the sentence with the centroid of each category
        distances = []
        for category in self.category_centroids:
            distances.append(acd_restaurants_train.cosine_similarity(normalized_centroid_feats, category)[0][0])

        stemmed_words = []
        stemmed_bi_words = []
        for w in words:
            if w not in stopwords:
                stemmed_words.append(self.stemmer.stem(w))
            stemmed_bi_words.append(self.stemmer.stem(w))
            
        stemmed_bigrams = nltk.bigrams(stemmed_bi_words)
        stemmed_bigrams_list = []
        for w in stemmed_bigrams:
            stemmed_bigrams_list.append(w)
                
        bigram_words = nltk.bigrams(words)
        bigram_list = []
        for w in bigram_words:
            bigram_list.append(w)

        tags = nltk.pos_tag(words)
        tags_set = set()
        for _, t in tags:
            tags_set.add(t)

        bitags = nltk.bigrams(list(tags_set))
        bitag_list = []
        for t in bitags:
            bitag_list.append(t)

        unigrams_feats = []
        bigrams_feats = []
        bipos_feats = []
        stemmed_unigrams_feats = []
        stemmed_bigrams_feats = []

        #unigrams features
        unigrams_feats = acd_restaurants_train.assign_features(self.unigrams_lexica, words, False)

        #bigrams features
        bigrams_feats = acd_restaurants_train.assign_features(self.bigrams_lexica, bigram_list, True)

        #pos bigrams features
        bipos_feats = acd_restaurants_train.assign_features(self.bipos_lexica,bitag_list, True)
                
        #stemmed_unigram features
        stemmed_unigrams_feats = acd_restaurants_train.assign_features(self.stemmed_unigrams_lexica, stemmed_words, False)

        #stemmed_bigram features
        stemmed_bigrams_feats = acd_restaurants_train.assign_features(self.stemmed_bigrams_lexica, stemmed_bigrams_list, True) 
                    
        test_sentences1.append(unigrams_feats + bigrams_feats + bipos_feats + stemmed_unigrams_feats + stemmed_bigrams_feats)
        test_sentences2.append(normalized_centroid_feats + distances)
            
        test_features1 = np.asarray(test_sentences1)
        test_features2 = np.asarray(test_sentences2)

        print('Done!')
        print('Predicting categories...')
        categories = []
        
        for i, test_fvector1 in enumerate(test_features1):
                #we get the [0,1] index, because on the [0,0] is the prediction for the category '0'
                food_pred1 = self.food_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                drinks_pred1 = self.drinks_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                service_pred1 = self.service_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                ambience_pred1 = self.ambience_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                location_pred1 = self.location_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                restaurant_pred1 = self.restaurant_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                general_pred1 = self.general_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                price_pred1 = self.price_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                quality_pred1 = self.quality_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                style_pred1 = self.style_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]
                misc_pred1 = self.misc_clf1.predict_proba(test_fvector1.reshape(1,-1))[0,1]

                food_pred2 = self.food_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                drinks_pred2 = self.drinks_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                service_pred2 = self.service_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                ambience_pred2 = self.ambience_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                location_pred2 = self.location_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                restaurant_pred2 = self.restaurant_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                general_pred2 = self.general_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                price_pred2 = self.price_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                quality_pred2 = self.quality_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                style_pred2 = self.style_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                misc_pred2 = self.misc_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]

                #dictionaries containing the probabilities for every E and A category
                entity_prob = {"food": (food_pred1+food_pred2)/2, "drinks": (drinks_pred1+drinks_pred2)/2,
                               "service": (service_pred1+service_pred2)/2, "ambience": (ambience_pred1+ambience_pred2)/2,
                                "location": (location_pred1+location_pred2)/2,
                               "restaurant": (restaurant_pred1+restaurant_pred2)/2}

                attr_prob = {"general": (general_pred1+general_pred2)/2, "prices": (price_pred1+price_pred2)/2,
                             "quality": (quality_pred1+quality_pred2)/2, "style_options": (style_pred1+style_pred2)/2,
                            "miscellaneous": (misc_pred1+misc_pred2)/2}

                sorted_entity_prob = sorted(entity_prob.items(), key=operator.itemgetter(1), reverse=True)
                sorted_attr_prob = sorted(attr_prob.items(), key=operator.itemgetter(1), reverse=True)
                
                for entity in sorted_entity_prob:
                    for attr in sorted_attr_prob:
                        if entity[1] > 0.4 and attr[1] > 0.4:
                            category = entity[0]+'#'+attr[0]
                            for c in self.cat_dict:
                                #if the e#a exists in the category dictionary and has > 0 appearances
                                if category == c and self.cat_dict[c] > 0:
                                    categories.append(category)
                                

        print('Done!')
        return categories
