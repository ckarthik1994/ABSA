author__ = 'ck'

import sys
import pymongo
from enum import Enum
import json
import os
#from ..mongodb_interface import mongodb_interface
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
#from mongodb_interface import mongodb_interface
import json
import math

uri1 = 'mongodb://Admin:qwerty123@ds261429.mlab.com:61429/textminingproject'
uri2 ='mongodb://Admin:qwerty123@ds211440.mlab.com:11440/textminingprojectdf'
from xml.dom import minidom
import json
import operator

def XMLParser():
    xmldoc = minidom.parse('ABSA16_Laptops_Train_SB1_v2.xml')
    reviewList = xmldoc.getElementsByTagName('Review')
    client = pymongo.MongoClient(uri1)
    db = client.get_default_database()
    db_name = db["TextMining"]

    client2 = pymongo.MongoClient(uri2)
    db2 = client2.get_default_database()
    db2_name = db2["TextMiningDF"]
    
    count_DF = {}
    for review in reviewList:
        reviewDict = dict()
        rid = review.attributes['rid'].value
        reviewText = ''
        opinions = set()
        count_TF = {}
        sentenceList = review.getElementsByTagName('sentence')
        unique_opinions = set()
        for sentence in sentenceList:
            textList = sentence.getElementsByTagName('text')
            for text in textList:
                for childNode in text.childNodes:
                    reviewText += childNode.nodeValue + ' '
            opinionList = review.getElementsByTagName('Opinion')
            for opinion in opinionList:
                
                cat = opinion.attributes['category'].value
                pol = opinion.attributes['polarity'].value

                opinion = '('+cat+','+pol+')'
                print opinion
                opinions.add(opinion)
                #TF
                if count_TF.has_key(opinion):
                    count = count_TF.get(opinion)
                    count+=1
                    count_TF[opinion] = count
                else:
                    count_TF[opinion] = 1
                unique_opinions.add(opinion)
            
        #DF
        for opinion in unique_opinions:
            if count_DF.has_key(opinion):
                count = count_DF.get(opinion)
                count+=1
                count_DF[opinion] = count
            else:
                count_DF[opinion] = 1
                unique_opinions.add(opinion)
            
        opinionsWithCount = set()
        #print opinions
        #print count_TF
        #print reviewText

        for opinion in opinions:
            count = count_TF[opinion]
            opinionWithCount = opinion[:-1]+','+str(count)+')'
            opinionsWithCount.add(opinionWithCount)
        ListToString = ':'.join(map(str, opinionsWithCount))
        reviewText = reviewText.strip()
        reviewDict['Opinions'] = ListToString
        reviewDict['Review'] = reviewText
        reviewDict['ReviewID'] = rid
        reviewJson = json.dumps(reviewDict, ensure_ascii=False)
        json_obj = json.loads(reviewJson)
        
        #write code to save to mongoDB
        db_name.insert(json_obj)
        
    for opinion in count_DF:
        #print opinion,count_DF[opinion]
        temp_string = '{"Opinion":"' + opinion + '", "DF" : "'+ str(count_DF[opinion]) +'"}'
        json_obj = json.loads(temp_string)
        db2_name.insert(json_obj)
        
    client.close()

#global_list_of_documents = []
def GetSimilarDocuments(query_opinions):
    client = pymongo.MongoClient(uri1)
    db = client.get_default_database()
    db_name = db["TextMining"]
    
    list_of_documents = []
    for Opinion in query_opinions:
        #print Opinion[1:-1]
        s = '{"Opinions" : {"$regex": "'+ Opinion[1:-1] +'"} }'
        json_obj = json.loads(s)
        cursor = db_name.find(json_obj)
        #cursor = db_name.find( {"Opinions" : {'$regex': "(LAPTOP#GENERAL,positive)"} } )
        for document in cursor:
            if document not in list_of_documents:
                list_of_documents.append(document)
                
    #print list_of_documents
    return list_of_documents

def CalculateIDF():
    client2 = pymongo.MongoClient(uri2)
    db2 = client2.get_default_database()
    db2_name = db2["TextMiningDF"]
    
    #DF
    IDF_dict = {}
    cursor_DF = db2_name.find({})
    for document in cursor_DF:
        IDF_dict[document['Opinion']]= float(1 + float(math.log(float(document['DF']))))
    #print IDF_dict
    return IDF_dict

def CalculateTF(document):
    TF_dict = {}

    list_of_opinions = document['Opinions'].split(':')
    for eachOpinion in list_of_opinions:
        opinionCategory = '('+eachOpinion.split(',')[0][1:]+ ',' +eachOpinion.split(',')[1]+')'
        polarity = eachOpinion.split(',')[-1][:-1]
        TF_dict[opinionCategory] = polarity
    #print TF_dict
    return TF_dict

def CosineSimilarity_TF_IDF(TF_values_Query,TF_values_document,IDF_values):
    dotProd = 0.0
    A = 0.0
    B = 0.0
    for Opinion in TF_values_Query:
        if(TF_values_document.has_key(Opinion)):
            dotProd+= (float(TF_values_document[Opinion])*float(IDF_values[Opinion])*float(TF_values_Query[Opinion])*float(IDF_values[Opinion]))

	if(TF_values_Query.has_key(Opinion) and IDF_values.has_key(Opinion) ) :
        	B += math.pow((float((TF_values_Query[Opinion])*float(IDF_values[Opinion]))),2)
    
    for Opinion in TF_values_document:
	if(TF_values_document.has_key(Opinion) and IDF_values.has_key(Opinion) ) :
        	A += math.pow((float(TF_values_document[Opinion])*float(IDF_values[Opinion])),2)

    if(A==0 or B==0):
        return 0

    CosineSimilarity = float(dotProd/(math.sqrt(A)*math.sqrt(B)))
    print CosineSimilarity
    return CosineSimilarity
    
def CosineSimilarity(list_of_documents,query_opinions):
    #print "CK"
    list_of_top_documents = {}
    ranked_results = []
    for document in list_of_documents:
        cnt=0
        for query in query_opinions:
            if query in document['Opinions']:
                cnt+=1
        list_of_top_documents[document['ReviewID']] = cnt
        
    sorted_list_of_top_documents = sorted(list_of_top_documents.items(), key=operator.itemgetter(1), reverse = True)
    print sorted_list_of_top_documents
    cnt=0
    for Review in sorted_list_of_top_documents:
        if cnt==5:
            break
        cnt+=1
        ranked_results.append(GetDocumentFromReviewID(Review[0]))
    return ranked_results




def GetDocumentFromReviewID(reviewID):
    client = pymongo.MongoClient(uri1)
    db = client.get_default_database()
    db_name = db["TextMining"]
    cursor = db_name.find( {"ReviewID" : reviewID } )

    for document in cursor:
        print "\n"
        print document
        print "\n"
        return document

def CalculateTFQuery(opinion_categories):
    count_TF={}
    for opinion in opinion_categories:
        #TF
        if count_TF.has_key(opinion):
            count = count_TF.get(opinion)
            count+=1
            count_TF[opinion] = count
        else:
            count_TF[opinion] = 1
    return count_TF
        
def GetTopDocumentsTFIDF(query_opinions):
    list_of_documents = GetSimilarDocuments(query_opinions)
    IDF_values = CalculateIDF()
    TF_values_Query = CalculateTFQuery(query_opinions)

    CosineSimilarity_dict ={}

    for document in list_of_documents:
        TF_values_document = CalculateTF(document)
        CosineSimilarity_document = CosineSimilarity_TF_IDF(TF_values_Query,TF_values_document,IDF_values)
        CosineSimilarity_dict[document['ReviewID']] = CosineSimilarity_document

    print CosineSimilarity_dict
    sorted_list_of_top_documents = sorted(CosineSimilarity_dict.items(), key=operator.itemgetter(1), reverse = True)
    print sorted_list_of_top_documents
    cnt=0
    for Review in sorted_list_of_top_documents:
        print Review
        if cnt==5:
            break
        cnt+=1
        GetDocumentFromReviewID(Review[0])

if __name__ == '__main__':
    #mdb_helper.InsertDocuments(test_json_new)
    #query_opinions = ["(u'LAPTOP#GENERAL', u'positive')","(u'LAPTOP#MISCELLANEOUS', u'positive')"]
    

    query_opinions = ["(LAPTOP#PRICE,positive)", "(LAPTOP#DESIGN_FEATURES,positive)","(HARD_DISC#DESIGN_FEATURES,positive)"]
    list_of_documents = GetSimilarDocuments(query_opinions)
    
    GetTopDocumentsTFIDF(query_opinions)
    
    #CosineSimilarity_TF_IDF(list_of_documents,query_opinions)
    #CosineSimilarity(list_of_documents,query_opinions)
    #mdb_helper.GetDocumentFromReviewID("B0074703CM_108_ANONYMOUS")
    
    #Insert into MongoDB from XML
    #XMLParser()
    


