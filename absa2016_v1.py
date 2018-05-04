#author Panagiotis Theodorakakos

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
import xml.etree.ElementTree as ET
import absa2016_lexicons
import absa2016_embeddings

#UNCOMMENT TO USE EMBEDDINGS MODEL
#import gensim
#from gensim.models import Word2Vec
#m = gensim.models.Word2Vec.load('model.bin') #load the model

#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
e = open("helpingEmbeddings.txt","r")
tmp = []
for a in e:
	tmp.append(a.split(" ")) #store the lexicon in an array
e.close()

m = []
tmpE = []
for line in tmp:
	for i in range(len(line)):
		l = line[i]
		if '[' in l:
			word = line[i-1]
			if len(l) != 1:
				tmpE.append(l[1:]) #handles cases where there is no space in the first cell
		elif ']' in l:
			l = l.replace('\n','').replace('\r','').replace(']','')
			if l !='' and l != ']':
				tmpE.append(l)
			
			tmpModel = []
			for i in range(len(tmpE)):
				if i != 0:
					tmpModel.append(float(tmpE[i])) #converting string to floats
			
			#convert to numpy array
			m.append([word,np.array(tmpModel)]) #append: word word_embeddings, skip the first cell of tempE containig the word
			tmpE = []
		else:
			l = l.replace('\n','').replace('\r','').replace(']','')
			if l !='' and l != ']':
				tmpE.append(l)
#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------


#============================================================================================================================


print '---------------- Laptops ----------------'
print
fold = 1
print('-------- Features Model--------')
fea2 = absa2016_lexicons.features('laptops/train_'+str(fold)+'.xml','laptops/test_'+str(fold)+'.xml','lap')
train_vector,train_tags = absa2016_lexicons.features.train(fea2,'lap')
absa2016_lexicons.features.trainModel(fea2, train_vector, train_tags, 'lap', fold)
print 'End version 5'

print('-------- Embeddings Model--------')
fea2 = absa2016_embeddings.features('laptops/train_'+str(fold)+'.xml','laptops/test_'+str(fold)+'.xml',m)
train_vector,train_tags = absa2016_embeddings.features.train(fea2)
absa2016_embeddings.features.trainModel(fea2, train_vector, train_tags, 'lap', fold)
print 'End version 6'
