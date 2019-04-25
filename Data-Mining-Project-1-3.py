import warnings
import numpy as np
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score, accuracy_score, auc
import matplotlib.pyplot as plt
import random
import numpy
import pylab
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
#stopwords for better results
stopwords = ENGLISH_STOP_WORDS

#open train_set.csv
train_data = pd.read_csv('train_set.csv', sep='\t')
category_criteria = "Content"
###

### initialize vectorizer ###
count_vect = CountVectorizer(stop_words=stopwords)
X_train_fit = count_vect.fit(train_data[category_criteria])
X_train_counts = count_vect.transform(train_data[category_criteria])
###




# RANDOM FOREST CLASSIFICATION
RANDOM_STATE = 123

### Converting Categories to Integers ####
category_dict = {'Politics':0 , 'Football':1, 'Business':2, 'Film':3, 'Technology':4}
categories = ["" for x in range(5)]
categories[0] = 'Politics'
categories[1] = 'Football'
categories[2] = 'Business'
categories[3] = 'Film'
categories[4] = 'Technology'
###

### transforming Categories to 0, 1, 2, 3, 4 ###
target = []
for x in train_data['Category']:
	target.append(category_dict[x])

target = numpy.array(target)
###

### LSI Usage: Components - Accuracy Graph ###
Components = [20, 50, 80, 110, 140, 170, 200]
accuracylist = []
for component in Components:
	svd =TruncatedSVD(n_components=component, random_state=42) #Dimensionality reduction using truncated SVD , reduces the reduntant data
	X_lsi =svd.fit_transform(X_train_counts)

	classifier = SGDClassifier()
	classifier.fit(X_lsi, target)
	yPred = classifier.predict(X_lsi)
	acc = accuracy_score(target, yPred)
	accuracylist.append(acc)

plt.title('Accuracy - Components')
plt.plot(Components, accuracylist)
plt.legend(loc = 'best')
plt.ylabel('Accuracy')
plt.xlabel('Components')
plt.show()
###


kf = KFold(n_splits=10)

### SVM ###
fold = 0
precision = 0.0
recall = 0.0
fm = 0.0
accuracy = 0.0
aucscore = 0.0
print("SVM")

for train_index, test_index in kf.split(train_data[category_criteria]):
	X_train_counts 	= count_vect.transform(train_data[category_criteria][train_index])
	X_test_counts = count_vect.transform(train_data[category_criteria][test_index].values.astype('U'))
		
	#classifier = svm.SVC(C=0.5, kernel='linear')
	classifier= svm.LinearSVC(multi_class = "ovr",random_state=RANDOM_STATE)
	classifier.fit(X_train_counts, target[train_index])
	yPred = classifier.predict(X_test_counts)
	prec= precision_score(target[test_index],yPred, average='macro')
	precision += prec
	rec = recall_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	recall += rec
	f_measure = f1_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	fm += f_measure
	acc = accuracy_score(target[test_index], yPred)
	accuracy += acc
	fpr, tpr, thresholds = metrics.roc_curve(target[test_index], yPred, pos_label=2)
	aucs= metrics.auc(fpr, tpr)
	aucscore += aucs
	fold += 1

	print ("Fold " + str(fold))
	print("Precision: ", prec)
	print ("Recall: ", rec)
	print("F-Measure: ", f_measure)
	print("Accuracy: ", acc)
	print ("AUC: ", aucs)

print('\n')

data1= numpy.empty(5, dtype=float)
data1[0] = accuracy /  10.0
data1[1] = precision / 10.0
data1[2] = recall / 10.0
data1[3] = fm / 10.0
data1[4] = aucscore / 10.0
###

### NAIVE BAYES ###
fold = 0
precision = 0.0
recall = 0.0
fm = 0.0
accuracy = 0.0
aucscore = 0.0
print("Naive Bayes")
i=0
for train_index, test_index in kf.split(train_data[category_criteria]):
	X_train_counts 	= count_vect.transform(train_data[category_criteria][train_index])
	X_test_counts = count_vect.transform(train_data[category_criteria][test_index].values.astype('U'))
		
	classifier = MultinomialNB()
	classifier.fit(X_train_counts, target[train_index])
	yPred = classifier.predict(X_test_counts)
	prec= precision_score(target[test_index],yPred, average='macro')
	precision += prec
	rec = recall_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	recall += rec
	f_measure = f1_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	fm += f_measure
	acc = accuracy_score(target[test_index], yPred)
	accuracy += acc
	fpr, tpr, thresholds = metrics.roc_curve(target[test_index], yPred, pos_label=2)
	aucs= metrics.auc(fpr, tpr)
	aucscore += aucs
	fold += 1

	print ("Fold " + str(fold))
	print("Precision: ", prec)
	print ("Recall: ", rec)
	print("F-Measure: ", f_measure)
	print("Accuracy: ", acc)
	print ("AUC: ", aucs)

print('\n')

data2= numpy.empty(5, dtype=float)
data2[0] = accuracy /  10.0
data2[1] = precision / 10.0
data2[2] = recall / 10.0
data2[3] = fm / 10.0
data2[4] = aucscore / 10.0
###

### RANDOM FOREST ###
fold = 0
precision = 0.0
recall = 0.0
fm = 0.0
accuracy = 0.0
aucscore = 0.0

print("Random Forest")

for train_index, test_index in kf.split(train_data[category_criteria]):
	X_train_counts 	= count_vect.transform(train_data[category_criteria][train_index])
	X_test_counts = count_vect.transform(train_data[category_criteria][test_index].values.astype('U'))

	classifier = RandomForestClassifier(warm_start=True, max_features="sqrt", random_state=RANDOM_STATE)
	classifier.fit(X_train_counts, target[train_index])
	yPred = classifier.predict(X_test_counts)
	prec= precision_score(target[test_index],yPred, average='macro')
	precision += prec
	rec = recall_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	recall += rec
	f_measure = f1_score(target[test_index], yPred, labels=None, pos_label=1, average='macro')
	fm += f_measure
	acc = accuracy_score(target[test_index], yPred)
	accuracy += acc
	fpr, tpr, thresholds = metrics.roc_curve(target[test_index], yPred, pos_label=2)
	aucs= metrics.auc(fpr, tpr)
	aucscore += aucs
	fold += 1

	print ("Fold " + str(fold))
	print("Precision: ", prec)
	print ("Recall: ", rec)
	print("F-Measure: ", f_measure)
	print("Accuracy: ", acc)
	print ("AUC: ", aucs)

print('\n')

data3= numpy.empty(5, dtype=float)
data3[0] = accuracy /  10.0
data3[1] = precision / 10.0
data3[2] = recall / 10.0
data3[3] = fm / 10.0
data3[4] = aucscore / 10.0
###


### open the test_set.csv
test_data = pd.read_csv('test_set.csv', sep='\t')
IDs=list(test_data['Id'])


### testSet_categories_RandomForest ###
X_train_fit = count_vect.fit(train_data[category_criteria])
X_train_counts = count_vect.transform(train_data[category_criteria])
classifier = SGDClassifier()
classifier.fit(X_train_counts, target)
X_test_counts = count_vect.transform(test_data[category_criteria].values.astype('U'))
prediction = classifier.predict(X_test_counts)

id =0
with open('testSet_categories_RandomForest.csv', 'w') as csvfile4:
	fieldnames = ['ID', 'Category']
	writer = csv.DictWriter(csvfile4, fieldnames = fieldnames)
	writer.writeheader()
	for x in prediction:
		writer.writerow({'ID': IDs[id], 'Category': categories[x]})
		id +=1
print('Created testSet_categories_RandomForest.csv')
### 

### testSet_categories_NaiveBayes ###
X_train_fit = count_vect.fit(train_data[category_criteria])
X_train_counts = count_vect.transform(train_data[category_criteria])
classifier = MultinomialNB()
classifier.fit(X_train_counts, target)
X_test_counts = count_vect.transform(test_data[category_criteria].values.astype('U'))
prediction = classifier.predict(X_test_counts)

id =0
with open('testSet_categories_NaiveBayes.csv', 'w') as csvfile4:
	fieldnames = ['ID', 'Category']
	writer = csv.DictWriter(csvfile4, fieldnames = fieldnames)
	writer.writeheader()
	for x in prediction:
		writer.writerow({'ID': IDs[id], 'Category': categories[x]})
		id +=1
print('Created testSet_categories_NaiveBayes.csv')
###

### testSet_categories_SVM ###
X_train_fit = count_vect.fit(train_data[category_criteria])
X_train_counts = count_vect.transform(train_data[category_criteria])
classifier = svm.LinearSVC(multi_class = "ovr",random_state=RANDOM_STATE)
classifier.fit(X_train_counts, target)
X_test_counts = count_vect.transform(test_data[category_criteria].values.astype('U'))
prediction = classifier.predict(X_test_counts)

id =0
with open('testSet_categories_SVM.csv', 'w') as csvfile4:
	fieldnames = ['ID', 'Category']
	writer = csv.DictWriter(csvfile4, fieldnames = fieldnames)
	writer.writeheader()
	for x in prediction:
		writer.writerow({'ID': IDs[id], 'Category': categories[x]})
		id +=1

print('Created testSet_categories_SVM.csv')
###

with open('EvalutionMetric_10fold.csv', 'w') as csvfile5:
	fieldnames = ['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM']
	writer = csv.DictWriter(csvfile5, fieldnames = fieldnames)
	writer.writeheader()
	for i in range(0, 5):
		if i==0:
			measure = 'Accuracy'
		elif i==1:
			measure = 'Precision'
		elif i==2:
			measure = 'Recall'
		elif i==3:
			measure = 'F-Measure'
		elif i==4:
			measure = 'AUC'
		writer.writerow({'Statistic Measure' : measure, 'Naive Bayes': data1[i], 'Random Forest': data2[i], 'SVM': data3[i]})

print('Created EvalutionMetric_10fold.csv')