#!/usr/bin/env python

'''
**Aspect Category Detection for the 5th task of SemEval 2016**
Unconstrained Submission for the Laptops domain

Run from the terminal:
>>> python acd_unconstrained_laptops.py --train train.xml --test test.xml
'''

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy, os, warnings
    import numpy as np
    from collections import Counter
    import operator
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.svm import SVC
    import nltk
    import os
    from nltk.stem import PorterStemmer
    from xml.sax.saxutils import escape
    from sklearn.externals import joblib
    import acd_laptops_train
except:
    sys.exit('Some package is missing... Perhaps <re>?')

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
	
fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')

class AspectCategoryClassifier:
    def __init__(self):
        self.stemmer = PorterStemmer()
        cats = joblib.load('./models/acd/lap/categories.pkl')
        self.cat_dict = Counter(cats)
        self.w2v_model = joblib.load('./models/acd/lap/w2v_model.pkl')

        self.unigrams_lexica = joblib.load('./models/acd/lap/unigrams_lexica.pkl')
        self.bigrams_lexica = joblib.load('./models/acd/lap/bigrams_lexica.pkl')
        self.bipos_lexica = joblib.load('./models/acd/lap/bipos_lexica.pkl')
        self.stemmed_unigrams_lexica = joblib.load('./models/acd/lap/stemmed_unigrams_lexica.pkl')
        self.stemmed_bigrams_lexica = joblib.load('./models/acd/lap/stemmed_bigrams_lexica.pkl')
        self.idf_dict = joblib.load('./models/acd/lap/idf_dict.pkl')
        self.category_centroids = joblib.load('./models/acd/lap/category_centroids.pkl')

        self.laptop_clf1 = joblib.load('./models/acd/lap/laptop_clf1Model.pkl')
        self.display_clf1 = joblib.load('./models/acd/lap/display_clf1Model.pkl')
        self.cpu_clf1 = joblib.load('./models/acd/lap/cpu_clf1Model.pkl')
        self.mb_clf1 = joblib.load('./models/acd/lap/mb_clf1Model.pkl')
        self.hd_clf1 = joblib.load('./models/acd/lap/hd_clf1Model.pkl')
        self.memory_clf1 = joblib.load('./models/acd/lap/memory_clf1Model.pkl')
        self.battery_clf1 = joblib.load('./models/acd/lap/battery_clf1Model.pkl')
        self.power_clf1 = joblib.load('./models/acd/lap/power_clf1Model.pkl')
        self.keyboard_clf1 = joblib.load('./models/acd/lap/keyboard_clf1Model.pkl')
        self.mouse_clf1 = joblib.load('./models/acd/lap/mouse_clf1Model.pkl')
        self.fans_clf1 = joblib.load('./models/acd/lap/fans_clf1Model.pkl')
        self.opt_drives_clf1 = joblib.load('./models/acd/lap/opt_drives_clf1Model.pkl')
        self.ports_clf1 = joblib.load('./models/acd/lap/ports_clf1Model.pkl')
        self.graphics_clf1 = joblib.load('./models/acd/lap/graphics_clf1Model.pkl')
        self.mm_devs_clf1 = joblib.load('./models/acd/lap/mm_devs_clf1Model.pkl')
        self.hardw_clf1 = joblib.load('./models/acd/lap/hardw_clf1Model.pkl')
        self.os_clf1 = joblib.load('./models/acd/lap/os_clf1Model.pkl')
        self.softw_clf1 = joblib.load('./models/acd/lap/softw_clf1Model.pkl')
        self.warranty_clf1 = joblib.load('./models/acd/lap/warranty_clf1Model.pkl')
        self.shipping_clf1 = joblib.load('./models/acd/lap/shipping_clf1Model.pkl')
        self.support_clf1 = joblib.load('./models/acd/lap/support_clf1Model.pkl')
        self.company_clf1 = joblib.load('./models/acd/lap/company_clf1Model.pkl')
        self.general_clf1 = joblib.load('./models/acd/lap/general_clf1Model.pkl')
        self.price_clf1 = joblib.load('./models/acd/lap/price_clf1Model.pkl')
        self.quality_clf1 = joblib.load('./models/acd/lap/quality_clf1Model.pkl')
        self.op_perf_clf1 = joblib.load('./models/acd/lap/op_perf_clf1Model.pkl')
        self.usability_clf1 = joblib.load('./models/acd/lap/usability_clf1Model.pkl')
        self.des_feats_clf1 = joblib.load('./models/acd/lap/des_feats_clf1Model.pkl')
        self.portability_clf1 = joblib.load('./models/acd/lap/portability_clf1Model.pkl')
        self.connectivity_clf1 = joblib.load('./models/acd/lap/connectivity_clf1Model.pkl')
        self.misc_clf1 = joblib.load('./models/acd/lap/misc_clf1Model.pkl')

        self.laptop_clf2 = joblib.load('./models/acd/lap/laptop_clf2Model.pkl')
        self.display_clf2 = joblib.load('./models/acd/lap/display_clf2Model.pkl')
        self.cpu_clf2 = joblib.load('./models/acd/lap/cpu_clf2Model.pkl')
        self.mb_clf2 = joblib.load('./models/acd/lap/mb_clf2Model.pkl')
        self.hd_clf2 = joblib.load('./models/acd/lap/hd_clf2Model.pkl')
        self.memory_clf2 = joblib.load('./models/acd/lap/memory_clf2Model.pkl')
        self.battery_clf2 = joblib.load('./models/acd/lap/battery_clf2Model.pkl')
        self.power_clf2 = joblib.load('./models/acd/lap/power_clf2Model.pkl')
        self.keyboard_clf2 = joblib.load('./models/acd/lap/keyboard_clf2Model.pkl')
        self.mouse_clf2 = joblib.load('./models/acd/lap/mouse_clf2Model.pkl')
        self.fans_clf2 = joblib.load('./models/acd/lap/fans_clf2Model.pkl')
        self.opt_drives_clf2 = joblib.load('./models/acd/lap/opt_drives_clf2Model.pkl')
        self.ports_clf2 = joblib.load('./models/acd/lap/ports_clf2Model.pkl')
        self.graphics_clf2 = joblib.load('./models/acd/lap/graphics_clf2Model.pkl')
        self.mm_devs_clf2 = joblib.load('./models/acd/lap/mm_devs_clf2Model.pkl')
        self.hardw_clf2 = joblib.load('./models/acd/lap/hardw_clf2Model.pkl')
        self.os_clf2 = joblib.load('./models/acd/lap/os_clf2Model.pkl')
        self.softw_clf2 = joblib.load('./models/acd/lap/softw_clf2Model.pkl')
        self.warranty_clf2 = joblib.load('./models/acd/lap/warranty_clf2Model.pkl')
        self.shipping_clf2 = joblib.load('./models/acd/lap/shipping_clf2Model.pkl')
        self.support_clf2 = joblib.load('./models/acd/lap/support_clf2Model.pkl')
        self.company_clf2 = joblib.load('./models/acd/lap/company_clf2Model.pkl')
        self.general_clf2 = joblib.load('./models/acd/lap/general_clf2Model.pkl')
        self.price_clf2 = joblib.load('./models/acd/lap/price_clf2Model.pkl')
        self.quality_clf2 = joblib.load('./models/acd/lap/quality_clf2Model.pkl')
        self.op_perf_clf2 = joblib.load('./models/acd/lap/op_perf_clf2Model.pkl')
        self.usability_clf2 = joblib.load('./models/acd/lap/usability_clf2Model.pkl')
        self.des_feats_clf2 = joblib.load('./models/acd/lap/des_feats_clf2Model.pkl')
        self.portability_clf2 = joblib.load('./models/acd/lap/portability_clf2Model.pkl')
        self.connectivity_clf2 = joblib.load('./models/acd/lap/connectivity_clf2Model.pkl')
        self.misc_clf2 = joblib.load('./models/acd/lap/misc_clf2Model.pkl')  

    def identifyQueryCategory(self, query):
        print('Creating test feature vectors...')
    
        test_sentences1 = []
        test_sentences2 = []

        words = (re.findall(r"[\w']+", query.lower())) #the unigrams list

        sentence_without_stopwords = ""
        for w in words:
            if w not in stopwords:
                sentence_without_stopwords = sentence_without_stopwords + " " + w
        clean_words = acd_laptops_train.clean(sentence_without_stopwords).split()

        #calculate the embedding of the words of the current sentence
        sentence_vector_feats = []
        words_with_embeds = []
        for w in clean_words:
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

        normalized_centroid_feats = acd_laptops_train.normalize_horizontal(centroid_feats)

        #compute the cosine similarity of the centroid of the sentence with the centroid of each category
        distances = []
        for category in self.category_centroids:
            distances.append(acd_laptops_train.cosine_similarity(normalized_centroid_feats, category)[0][0])

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
        unigrams_feats = acd_laptops_train.assign_features(self.unigrams_lexica, words, False)

        #bigrams features
        bigrams_feats = acd_laptops_train.assign_features(self.bigrams_lexica, bigram_list, True)

        #pos bigrams features
        bipos_feats = acd_laptops_train.assign_features(self.bipos_lexica,bitag_list, True)
                
        #stemmed_unigram features
        stemmed_unigrams_feats = acd_laptops_train.assign_features(self.stemmed_unigrams_lexica, stemmed_words, False)

        #stemmed_bigram features
        stemmed_bigrams_feats = acd_laptops_train.assign_features(self.stemmed_bigrams_lexica, stemmed_bigrams_list, True) 
                    
        test_sentences1.append(unigrams_feats + bigrams_feats + bipos_feats + stemmed_unigrams_feats + stemmed_bigrams_feats)
        test_sentences2.append(normalized_centroid_feats + distances)
        
        test_features1 = np.asarray(test_sentences1)
        test_features2 = np.asarray(test_sentences2)
        aspect_categories = []

        print('Done!')
        print('Predicting categories...')
        categories = []

        for i, test_fvector in enumerate(test_features1):     
                laptop_pred1 = self.laptop_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                display_pred1 = self.display_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                cpu_pred1 = self.cpu_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                mb_pred1 = self.mb_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                hd_pred1 = self.hd_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                memory_pred1 = self.memory_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                battery_pred1 = self.battery_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                power_pred1 = self.power_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                keyboard_pred1 = self.keyboard_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                mouse_pred1 = self.mouse_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                fans_pred1 = self.fans_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                opt_drives_pred1 = self.opt_drives_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                ports_pred1 = self.ports_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                graphics_pred1 = self.graphics_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                mm_devs_pred1 = self.mm_devs_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                hardw_pred1 = self.hardw_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                os_pred1 = self.os_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                softw_pred1 = self.softw_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                warranty_pred1 = self.warranty_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                shipping_pred1 = self.shipping_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                support_pred1 = self.support_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                company_pred1 = self.company_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                general_pred1 = self.general_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                price_pred1 = self.price_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                quality_pred1 = self.quality_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                op_perf_pred1 = self.op_perf_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                usability_pred1 = self.usability_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                des_feats_pred1 = self.des_feats_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                portability_pred1 = self.portability_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                connectivity_pred1 = self.connectivity_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]
                misc_pred1 = self.misc_clf1.predict_proba(test_fvector.reshape(1,-1))[0,1]

                laptop_pred2 = self.laptop_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                display_pred2 = self.display_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                cpu_pred2 = self.cpu_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                mb_pred2 = self.mb_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                hd_pred2 = self.hd_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                memory_pred2 = self.memory_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                battery_pred2 = self.battery_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                power_pred2 = self.power_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                keyboard_pred2 = self.keyboard_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                mouse_pred2 = self.mouse_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                fans_pred2 = self.fans_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                opt_drives_pred2 = self.opt_drives_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                ports_pred2 = self.ports_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                graphics_pred2 = self.graphics_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                mm_devs_pred2 = self.mm_devs_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                hardw_pred2 = self.hardw_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                os_pred2 = self.os_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                softw_pred2 = self.softw_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                warranty_pred2 = self.warranty_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                shipping_pred2 = self.shipping_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                support_pred2 = self.support_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                company_pred2 = self.company_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                general_pred2 = self.general_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                price_pred2 = self.price_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                quality_pred2 = self.quality_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                op_perf_pred2 = self.op_perf_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                usability_pred2 = self.usability_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                des_feats_pred2 = self.des_feats_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                portability_pred2 = self.portability_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                connectivity_pred2 = self.connectivity_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]
                misc_pred2 = self.misc_clf2.predict_proba(test_features2[i].reshape(1,-1))[0,1]

                entity_prob = {"laptop": (laptop_pred1+laptop_pred2)/2, "display": (display_pred1+display_pred2)/2, "cpu": (cpu_pred1+cpu_pred2)/2,
                                    "motherboard": (mb_pred1+mb_pred2)/2, "hard_disc": (hd_pred1+hd_pred2)/2,
                                   "memory": (memory_pred1+memory_pred2)/2, "battery": (battery_pred1+battery_pred2)/2,
                                   "power_supply": (power_pred1+power_pred2)/2, "keyboard": (keyboard_pred1+keyboard_pred2)/2,
                                   "mouse": (mouse_pred1+mouse_pred2)/2,
                                   "fans_cooling": (fans_pred1+fans_pred2)/2, "optical_drives": (opt_drives_pred1+opt_drives_pred2)/2,
                                    "ports": (ports_pred1+ports_pred2)/2, "graphics": (graphics_pred1+graphics_pred2)/2,
                                   "multimedia_devices": (mm_devs_pred1+mm_devs_pred2)/2, "hardware": (hardw_pred1+hardw_pred2)/2,
                                   "os": (os_pred1+os_pred2)/2, "software": (softw_pred1+softw_pred2)/2, "warranty": (warranty_pred1+warranty_pred2)/2,
                                   "shipping": (shipping_pred1+shipping_pred2)/2, "support": (support_pred1+support_pred2)/2,
                                   "company": (company_pred1+company_pred2)/2}

                attr_prob = {"general": (general_pred1+general_pred2)/2, "price": (price_pred1+price_pred2)/2, "quality": (quality_pred1+quality_pred2)/2,
                                 "operation_performance": (op_perf_pred1+op_perf_pred2)/2, "usability": (usability_pred1+usability_pred2)/2,
                                 "design_features": (des_feats_pred1+des_feats_pred2)/2,
                                 "portability": (portability_pred1+portability_pred2)/2, "connectivity": (connectivity_pred1+connectivity_pred2)/2,
                                 "miscellaneous": (misc_pred1+misc_pred2)/2}

                sorted_entity_prob = sorted(entity_prob.items(), key=operator.itemgetter(1), reverse=True)
                sorted_attr_prob = sorted(attr_prob.items(), key=operator.itemgetter(1), reverse=True)
                
                
                for entity in sorted_entity_prob:
                    for attr in sorted_attr_prob:
                        if entity[1] > 0.4 and attr[1] > 0.4:
                            category = entity[0]+'#'+attr[0]
                            for c in self.cat_dict:
                                if category == c and self.cat_dict[c] > 0:
                                    categories.append(category.upper())
                                
                    
        print('Done!')
        return categories