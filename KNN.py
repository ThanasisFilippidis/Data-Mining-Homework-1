import csv
import random
import math
import operator
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold


def euclideanDistance(spot1, spot2, length):
	distance = 0
	for x in range(length):
		distance += pow((spot1[x] - spot2[x]), 2)
	return math.sqrt(distance)

def findNeighbors(trainingData, testData, k):
	distances = []
	length = len(testData)-1
	for x in range(len(trainingData)):
		dist = euclideanDistance(testData, trainingData[x], length)
		distances.append((trainingData[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	stopwords = ENGLISH_STOP_WORDS

	#open train_set.csv
	train_data = pd.read_csv('train_set.csv', sep='\t')
	train_data = train_data.head(2000)
	category_criteria = "Content"

	#initialize vectorizer
	count_vect = CountVectorizer(stop_words=stopwords)
	X_train_fit = count_vect.fit(train_data[category_criteria])
	X_train_counts = count_vect.transform(train_data[category_criteria])

	#use LSI
	svd=TruncatedSVD(n_components=200, random_state=42) #Dimensionality reduction using truncated SVD , reduces the reduntant data
	X_train_counts=svd.fit_transform(X_train_counts)


	predictions=[]
	k = 3
	#n_splits = 10
	#kf = KFold(n_splits=n_splits,shuffle=True)
	#for train_set , test_set in kf.split(X_train_counts):
	#	train_set=train_set[:1000]
	#	test_set=test_set[:1000]
	#	for x in test_set:
	#		neighbors = findNeighbors(X_train_counts, X_train_counts[x], k)
	#		result = getResponse(neighbors)
	#		predictions.append(result)
	#	accuracy = getAccuracy(X_train_counts[test_set], predictions)
	#	print('Accuracy: ' + repr(accuracy) + '%')




	# generate predictions
	
	for x in range(len(X_train_counts)):
		neighbors = findNeighbors(X_train_counts, X_train_counts[x], k)
		result = getResponse(neighbors)
		#print(result)
		predictions.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(X_train_counts[x][-1]))
	accuracy = getAccuracy(X_train_counts, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()