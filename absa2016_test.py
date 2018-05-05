import numpy as np
import codecs
import xml.etree.ElementTree as ET
import absa2016_lexicons
import absa2016_embeddings
from sklearn.externals import joblib
import nltk
import re

class SentimentClassifier:
	#weights for each method. each method's vote weights the same
	w1 = 0.5
	w2 = 0.5

	#embeddings model
	m = []

	def __init__(self):
		#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
		e = open("helpingEmbeddings.txt","r")
		tmp = []
		for a in e:
			tmp.append(a.split(" ")) #store the lexicon in an array
		e.close()

		self.m = []
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
					self.m.append([word,np.array(tmpModel)]) #append: word word_embeddings, skip the first cell of tempE containig the word
					tmpE = []
				else:
					l = l.replace('\n','').replace('\r','').replace(']','')
					if l !='' and l != ']':
						tmpE.append(l)
		#---------- use the helpingEmbeddings.txt instead of the embeddings model ----------

		print('-------- Features Model--------')
		self.fea2LaptopLexicons = absa2016_lexicons.features('','','lap')
		self.laptopLexiconsModel = joblib.load("models/polarity_detection/lap_lexiconModel.pkl")

		self.fea2RestaurantLexicons = absa2016_lexicons.features('','','rest')
		self.restaurantLexiconsModel = joblib.load("models/polarity_detection/rest_lexiconModel.pkl")

		print('-------- Embeddings Model--------')
		self.fea2Embeddings = absa2016_embeddings.features('','',self.m)
		self.laptopEmbeddingsModel = joblib.load("models/polarity_detection/lap_embeddingsModel.pkl")
		self.restaurantEmbeddingsModel = joblib.load("models/polarity_detection/rest_embeddingsModel.pkl")


		#============================================================================================================================

	def test(self, query, acd, domain):
		sentences = re.split('\.|,|;', query)
		subsentences = []
		for sentence in sentences:
			tokens = nltk.tokenize.word_tokenize(sentence)
			postags = nltk.pos_tag(tokens)
			subsentence = ''
			for postag in postags:
				if (postag[1]=='CC' and (postag[0].lower()=='or' or postag[0].lower()=='and' or postag[0].lower()=='but')) or (postag[1]=='IN' and postag[0].lower()=='with'):
					subsentence = subsentence.strip()
					subsentences.append(subsentence)
					subsentence = ''
				else:
					subsentence += postag[0] + ' '
			if not subsentence == '':
				subsentence = subsentence.strip()
				subsentences.append(subsentence)

		opinionCategories = []
		polarities = []

		for subsentence in subsentences:
			if subsentence == '':
				continue
			sent_categories, sent_polarities = self.test2(subsentence, acd, domain)
			for sent_category in sent_categories:
				opinionCategories.append(sent_category)
			for sent_polarity in sent_polarities:
				polarities.append(sent_polarity)

		return opinionCategories, polarities

	def test2(self, sentence, acd, domain):
		opinionCategories = acd.identifyQueryCategory(sentence)
		if len(opinionCategories)==0:
			return [],[]
		#print 'Opinion Category: ', opinionCategories
		#print '---------------- Laptops ----------------'
		#print
		
		if domain == 'lap':
			fea2Lexicons = self.fea2LaptopLexicons
			lexiconsModel = self.laptopLexiconsModel
			embeddingsModel = self.laptopEmbeddingsModel
		else:
			fea2Lexicons = self.fea2RestaurantLexicons
			lexiconsModel = self.restaurantLexiconsModel
			embeddingsModel = self.restaurantEmbeddingsModel

		test_vector = absa2016_lexicons.features.getTestVector(fea2Lexicons, sentence, opinionCategories, domain)
		predictionsLap1 = lexiconsModel.predict_proba(test_vector)

		test_vector = absa2016_embeddings.features.getTestVector(self.fea2Embeddings, sentence, opinionCategories)
		predictionsLap2 = embeddingsModel.predict_proba(test_vector)

		#both methods "vote"
		l = len(predictionsLap1)
		predictionsLap = []
		for i in range(l):
			a = float(predictionsLap1[i][0]*self.w1 + predictionsLap2[i][0]*self.w2)/2 #number of the methods we are using
			b = float(predictionsLap1[i][1]*self.w1 + predictionsLap2[i][1]*self.w2)/2
			c = float(predictionsLap1[i][2]*self.w1 + predictionsLap2[i][2]*self.w2)/2
			
			if a > b and a > c:
				predictionsLap.append('negative') #check the probabilities
			elif b > a and b > c:
				predictionsLap.append('neutral')
			elif c > a and c > b:
				predictionsLap.append('positive')

		return opinionCategories, predictionsLap

if __name__ == "__main__":
	query = "You can't get any better than this price and it come with an internal disk drive."
	#acd = AspectCategoryClassifier()
	#classifier = SentimentClassifier()
	#opinionCategories, polarity = classifier.test(query, acd, 'lap')
	#print polarity
