__author__ = 'ck'

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

test_json = '{"glossary": {"title": "example glossary","GlossDiv": { "title": "S","GlossList": {"GlossEntry": {"ID": "SGML",	"SortAs": "SGML",					"GlossTerm": "Standard Generalized Markup Language",					"Acronym": "SGML",					"Abbrev": "ISO 8879:1986",					"GlossDef": {                        "para": "A meta-markup language, used to create markup languages such as DocBook.",						"GlossSeeAlso": ["GML", "XML"]                    },					"GlossSee": "markup"                }            }        }    }'

test_json = '{  "Reviews": {    "Review": [      {        "rid": "B0074703CM_108_ANONYMOUS",        "sentences": {          "sentence": [            {              "id": "B0074703CM_108_ANONYMOUS:0",              "text": "Well, my first apple computer and I am impressed.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#GENERAL",                  "polarity": "positive"                }              }            },            {              "id": "B0074703CM_108_ANONYMOUS:1",              "text": "Works well, fast and no reboots.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#OPERATION_PERFORMANCE",                  "polarity": "positive"                }              }            },            {              "id": "B0074703CM_108_ANONYMOUS:4",              "text": "Glad I did so far.",              "Opinions": {                "Opinion": [                  {                    "category": "COMPANY#GENERAL",                    "polarity": "positive"                  },                  {                    "category": "LAPTOP#GENERAL",                    "polarity": "positive"                  }                ]              }            },            {              "id": "B0074703CM_108_ANONYMOUS:5",              "text": "Laptop is good but gets heated very fast.",              "Opinions": {                "Opinion": [                  {                    "category": "laptop#quality",                    "polarity": "positive"                  },                  {                    "category": "laptop#operation_performance",                    "polarity": "positive"                  }                ]              }            }          ]        }      },      {        "rid": "B00GJUQ4Z0_10_ANONYMOUS",        "sentences": {          "sentence": [            {              "id": "B00GJUQ4Z0_10_ANONYMOUS:0",              "text": "s.... L .... o..... w.... rea......llllyy slow.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#OPERATION_PERFORMANCE",                  "polarity": "positive"                }              }            },            {              "id": "B00GJUQ4Z0_10_ANONYMOUS:1",              "text": "like seriously  really slow.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#OPERATION_PERFORMANCE",                  "polarity": "negative"                }              }            },            {              "id": "B00GJUQ4Z0_10_ANONYMOUS:2",              "text": "impossible to use.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#USABILITY",                  "polarity": "negative"                }              }            },            {              "id": "B00GJUQ4Z0_10_ANONYMOUS:3",              "text": "cant even read properly.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#USABILITY",                  "polarity": "negative"                }              }            },            {              "id": "B00GJUQ4Z0_10_ANONYMOUS:4",              "text": "plus  no russian input ?? wtf",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#MISCELLANEOUS",                  "polarity": "negative"                }              }            }          ]        }      },      {        "rid": "B0146DD02G_18_ANONYMOUS",        "sentences": {          "sentence": [            {              "id": "B0146DD02G_18_ANONYMOUS:0",              "text": "What a great laptop, I can run my games and work really fast.",              "Opinions": {                "Opinion": [                  {                    "category": "LAPTOP#GENERAL",                    "polarity": "positive"                  },                  {                    "category": "LAPTOP#MISCELLANEOUS",                    "polarity": "positive"                  },                  {                    "category": "LAPTOP#OPERATION_PERFORMANCE",                    "polarity": "positive"                  }                ]              }            },            {              "id": "B0146DD02G_18_ANONYMOUS:1",              "text": "Really light you can carry with you everywhere.",              "Opinions": {                "Opinion": [                  {                    "category": "LAPTOP#DESIGN_FEATURES",                    "polarity": "positive"                  },                  {                    "category": "LAPTOP#PORTABILITY",                    "polarity": "positive"                  }                ]              }            },            {              "id": "B0146DD02G_18_ANONYMOUS:2",              "text": "Great battery life.",              "Opinions": {                "Opinion": {                  "category": "BATTERY#OPERATION_PERFORMANCE",                  "polarity": "positive"                }              }            },            {              "id": "B0146DD02G_18_ANONYMOUS:3",              "text": "Everything at a very great price.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#PRICE",                  "polarity": "positive"                }              }            },            {              "id": "B0146DD02G_18_ANONYMOUS:4",              "text": "I completely recommend it.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#GENERAL",                  "polarity": "positive"                }              }            }          ]        }      }    ]  } }'


test_json_new = '{  "Reviews": {    "Review": [      {        "rid": "B0074703CM_108_ANONYMOUS",        "sentences": {          "sentence": [            {              "id": "B0074703CM_108_ANONYMOUS:0",              "text": "Well, my first apple computer and I am impressed.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#GENERAL",                  "polarity": "positive"                }              }            },            {              "id": "B0074703CM_108_ANONYMOUS:1",              "text": "Works well, fast and no reboots.",              "Opinions": {                "Opinion": {                  "category": "LAPTOP#OPERATION_PERFORMANCE",                  "polarity": "positive"                }              }            },            {              "id": "B0074703CM_108_ANONYMOUS:4",              "text": "Glad I did so far.",              "Opinions": {                "Opinion": [                  {                    "category": "COMPANY#GENERAL",                    "polarity": "positive"                  },                  {                    "category": "LAPTOP#GENERAL",                    "polarity": "positive"                  }                ]              }            },            {              "id": "B0074703CM_108_ANONYMOUS:5",              "text": "Laptop is good but gets heated very fast.",              "Opinions": {                "Opinion": [                  {                    "category": "laptop#quality",                    "polarity": "positive"                  },                  {                    "category": "laptop#operation_performance",                    "polarity": "positive"                  }                ]              }            }          ]        }      },      {        "rid": "B00GJUQ4Z0_10_ANONYMOUS",        "sentences": {          "sentence": {            "id": "B00GJUQ4Z0_10_ANONYMOUS:0",            "text": "s.... L .... o..... w.... rea......llllyy slow.",            "Opinions": {              "Opinion": {                "category": "LAPTOP#OPERATION_PERFORMANCE",                "polarity": "positive"              }            }          }        }      }    ]  } }'

uri = 'mongodb://Admin:qwerty123@ds261429.mlab.com:61429/textminingproject'

class mongodb_helper():

    def InsertDocuments(self,query_json_obj):
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        db_name = db["TextMining"]

        
        LoadedJson = json.loads(query_json_obj)['Reviews']['Review']
        print LoadedJson
        print type(LoadedJson)
        #print RawJsonToJsonObj
        #JsonObj = json.dumps(LoadedJson)

        #print JsonObj['Review']
        for review in LoadedJson:
            #print "working with a tweet, bound to variable res3"
            #print review.keys()
            #print review
            reviewID =  review['rid']
            #print type(review)
            sentences = review['sentences']
            #print type(sentences)
            
            CompleteReview = ""
            SetOfOpinions = []
            
            print "Type of sentence : " + str(type(sentences['sentence']))

            if( type(sentences['sentence']) is dict ):
                CompleteReview+= sentences['sentence']['text']
                if(type(sentences['sentence']['Opinions']) is dict):
                    Opinions = sentences['sentence']['Opinions']['Opinion']
                    temp_tuple = tuple((Opinions['category'],Opinions['polarity']))
                    if(temp_tuple not in SetOfOpinions):
                        SetOfOpinions.append(temp_tuple)
                else:
                    ListOfopinions =  sentences['sentence']['Opinions']['Opinion']
                    
                    for eachOpinion in ListOfopinions:
                        temp_tuple = tuple((eachOpinion['category'],eachOpinion['polarity']))
                        if(temp_tuple not in SetOfOpinions):
                            SetOfOpinions.append(temp_tuple)
                print sentences['sentence']['Opinions']
            else:   
                for sentence in sentences['sentence']:
                    if(type(sentence['Opinions']['Opinion']) is dict):
                        #print sentence
                        Opinions = sentence['Opinions']['Opinion']
                        temp_tuple = tuple((Opinions['category'],Opinions['polarity']))
                        if(temp_tuple not in SetOfOpinions):
                            SetOfOpinions.append(temp_tuple)
                        #print Opinions['category']
                        #print Opinions['polarity']
                    else:
                        ListOfopinions =  sentence['Opinions']['Opinion']
                        
                        for eachOpinion in ListOfopinions:
                            #print eachOpinion
                            temp_tuple = tuple((eachOpinion['category'],eachOpinion['polarity']))
                            #print temp_tuple
                            if(temp_tuple not in SetOfOpinions):
                                SetOfOpinions.append(temp_tuple)
                            
                    #print "Final Sentences"
                    #print sentence['text']
                    CompleteReview = CompleteReview + sentence['text'] + ' '
            
            print CompleteReview
            print SetOfOpinions
            print reviewID
            ListToString = ','.join(map(str, SetOfOpinions))
            print ListToString
            s = '{"ReviewID":"' + reviewID + '","Review":"'+CompleteReview+'","Opinions":"'+ListToString+'"}'
            json_obj = json.loads(s)
            db_name.insert(json_obj)
        client.close()

    global_list_of_documents = []
    def GetSimilarDocuments(self,query_opinions):
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        db_name = db["TextMining"]
        
        #query_string ='{ "$and" : [ {\"Opinions\" : {\'$regex\': \"(u\'LAPTOP#GENERAL\', u\'positive\')\"} },{\"Opinions\": {\'$regex\': \"(u\'LAPTOP#MISCELLANEOUS\', u\'positive\')\"} } ] }'
        #json_obj = json.loads(query_string)
        #print query_string

        list_of_documents = []
        for Opinion in query_opinions:
            print Opinion
            cursor = db_name.find( {"Opinions" : {'$regex': "(u'LAPTOP#GENERAL', u'positive')"} } )

            for document in cursor:
                if document not in list_of_documents:
                    list_of_documents.append(document)
                    #print "\n"
                    #print document
        
        print list_of_documents
        global global_list_of_documents 
        global_list_of_documents = list_of_documents

    global_list_of_top_documents = {}

    def CosineSimilarity(self,query_opinions):
        print "CK"
        list_of_top_documents = {}
        for document in global_list_of_documents:
            cnt=0
            for query in query_opinions:
                if query in document['Opinions']:
                    cnt+=1
            print document
            list_of_top_documents[document['ReviewID']] = cnt
            if cnt == len(query_opinions):
                print document

        global global_list_of_top_documents
        global_list_of_top_documents = list_of_top_documents
        print global_list_of_top_documents

#       cursor = db_name.find({"$and" : [ {"Opinions" : {'$regex': "(u'LAPTOP#GENERAL', u'positive')"} } ,                                         {"Opinions" : {'$regex': "(u'LAPTOP#MISCELLANEOUS', u'positive')"} } ] }          )
        #cursor = db_name.find(json_obj)
    
    def GetDocumentFromReviewID(self,reviewID):
        client = pymongo.MongoClient(uri)
        db = client.get_default_database()
        db_name = db["TextMining"]
        cursor = db_name.find( {"ReviewID" : reviewID } )

        for document in cursor:
            print document
            return document

if __name__ == '__main__':
    mdb_helper = mongodb_helper()
    #mdb_helper.InsertDocuments(test_json_new)
    #query_opinions = ["(u'LAPTOP#GENERAL', u'positive')","(u'LAPTOP#MISCELLANEOUS', u'positive')"]
    #mdb_helper.GetSimilarDocuments(query_opinions)
    #print mdb_helper.global_list_of_documents
    #mdb_helper.CosineSimilarity(query_opinions)
    mdb_helper.GetDocumentFromReviewID("B0074703CM_108_ANONYMOUS")
    
    
    #print mdb_helper.get_points_data("rice_pt_soda")
    #mdb_helper.recursive_file_read("pressure")
    #print mdb_helper.get_timeseries_data("2 Mag CHW Return Temp")

