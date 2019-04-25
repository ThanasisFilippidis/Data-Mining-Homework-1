import pandas as pd
import difflib
import numpy
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from random import uniform
from random import randrange
from scipy.sparse import dia_matrix
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

vectorizer = TfidfVectorizer(max_df=0.6, max_features=1000,
								 min_df=0.1, stop_words='english',
								 use_idf=True); #ftiaxnei enan pinaka o opios periexei se morfi dianusmatos kai exei mia mia tis lekseis pou periexontai sta keimena kai poses fores emfanizontai se kathe arthro
df = pd.read_csv("train_set.csv", sep = "\t"); #diavazontas ap to csv
rows=len(df)
print ("Number of file rows:")
print (rows)
Categories = list(df['Category']);
cont=list(df['Content'][:rows])
print ("Starting the .fit_transform")
Alfa =vectorizer.fit_transform(cont);
print ("fit.transform ended.")
Contents=Alfa.toarray();
linecounter = 0;
for line in Contents:
	linecounter+=1;
### 5 random centers ###
randomcenter=randrange(0,linecounter);
firstcen1=Contents[randomcenter][:];
randomcenter=randrange(0,linecounter);
firstcen2=Contents[randomcenter][:];
randomcenter=randrange(0,linecounter);
firstcen3=Contents[randomcenter][:];
randomcenter=randrange(0,linecounter);
firstcen4=Contents[randomcenter][:];
randomcenter=randrange(0,linecounter);
firstcen5=Contents[randomcenter][:];
### make it 1-D ###
firstcen1=firstcen1.reshape(1,-1);
firstcen2=firstcen2.reshape(1,-1);
firstcen3=firstcen3.reshape(1,-1);
firstcen4=firstcen4.reshape(1,-1);
firstcen5=firstcen5.reshape(1,-1);
cen1=firstcen1;
cen2=firstcen2;
cen3=firstcen3;
cen4=firstcen4;
cen5=firstcen5;
###


### we check if the centroids stopped moving
stopped1=False;
stopped2=False;
stopped3=False;
stopped4=False;
stopped5=False;
### max efforts = 100 ###
for i in range(0, 100): 
	print ("Effort No.",i,":")
	
	### initialize empty clusters ###
	cat1=[];
	cat2=[];
	cat3=[];
	cat4=[];
	cat5=[];
	cluster1=[];
	cluster2=[];
	cluster3=[];
	cluster4=[];
	cluster5=[];
	for j in range(linecounter):
		### calculate the distance of the content with each centre ###
		wowdistance1=1-cosine_similarity(firstcen1,Contents[j][:]);
		wowdistance2=1-cosine_similarity(firstcen2,Contents[j][:]);
		wowdistance3=1-cosine_similarity(firstcen3,Contents[j][:]);
		wowdistance4=1-cosine_similarity(firstcen4,Contents[j][:]);
		wowdistance5=1-cosine_similarity(firstcen5,Contents[j][:]);
		suchmin=min(wowdistance1,wowdistance2,wowdistance3,wowdistance4,wowdistance5);
		### append to the smallest distance ###
		if(suchmin==wowdistance1):
			cluster1.append(Contents[j][:]);
			cat1.append(Categories[j]);
		elif (suchmin==wowdistance2):
			cluster2.append(Contents[j][:]);
			cat2.append(Categories[j]);
		elif (suchmin==wowdistance3):
			cluster3.append(Contents[j][:]);
			cat3.append(Categories[j]);
		elif (suchmin==wowdistance4):
			cluster4.append(Contents[j][:]);
			cat4.append(Categories[j]);
		elif (suchmin==wowdistance5):
			cluster5.append(Contents[j][:]);
			cat5.append(Categories[j]);
	### save the previous centroids
	cen1=firstcen1
	cen2=firstcen2
	cen3=firstcen3
	cen4=firstcen4
	cen5=firstcen5
	### generate the new centroids
	firstcen1=numpy.mean(cluster1, axis=0);
	firstcen2=numpy.mean(cluster2, axis=0);
	firstcen3=numpy.mean(cluster3, axis=0);
	firstcen4=numpy.mean(cluster4, axis=0);
	firstcen5=numpy.mean(cluster5, axis=0);
	check1=False
	check2=False
	check3=False
	check4=False
	check5=False
	if (i>1):
		if (numpy.allclose(firstcen1,cen1)):
			check1=True
		if (numpy.allclose(firstcen2,cen2)):
			check2=True
		if (numpy.allclose(firstcen3,cen3)):
			check3=True
		if (numpy.allclose(firstcen4,cen4)):
			check4=True
		if (numpy.allclose(firstcen5,cen5)):
			check5=True
	
	print ("Centroid No.1 didn't move.",check1)
	print ("Centroid No.2 didn't move.",check2)
	print ("Centroid No.3 didn't move.",check3)
	print ("Centroid No.4 didn't move.",check4)
	print ("Centroid No.5 didn't move.",check5)
	if (check1 and check2 and check3 and check3 and check4 and check5):
		break;

n=5
polpercent = numpy.empty(n, dtype=float)
filmpercent = numpy.empty(n, dtype=float)
buspercent = numpy.empty(n, dtype=float)
footpercent = numpy.empty(n, dtype=float)
techpercent = numpy.empty(n, dtype=float)

for i in range(0, 5):

	#cluster1
	if i ==0:
		category =cat1
	elif i==1:
		category =cat2
	elif i==2:
		category=cat3
	elif i==3:
		category =cat4
	elif i==4:
		category = cat5
	pol=0
	film=0
	foot=0
	bus=0
	tect=0
	polp=0.0
	filmp=0.0
	footp=0.0
	busp=0.0
	techp=0.0
   
	for j in range(len(category)):
		if (category[j] == 'Politics'):
			pol+=1;
		elif (category[j] == 'Film'):
			film+=1;
		elif (category[j] == 'Football'):
			foot+=1;
		elif (category[j] == 'Business'):
			bus+=1;
		elif (category[j] == 'Technology'):
			tect+=1;

	if (len(category)!=0):
		### stats per cluster ###
		polp=float(pol)/float(len(category));
		filmp=float(film)/float(len(category));
		footp=float(foot)/float(len(category));
		busp=float(bus)/float(len(category));
		techp=float(tect)/float(len(category));
	else:
		polp=0.0
		filmp=0.0
		footp=0.0
		busp=0.0
		techp=0.0
	print ("Cluster ",i+1," was created.")
	polpercent[i] = round(polp,1)
	filmpercent[i] = round(filmp,1)
	footpercent[i] = round(footp,1)
	buspercent[i] = round(busp,1)
	techpercent[i] = round(techp,1)

### create the csv file ###
with open('clustering_KMeans.csv', 'w') as csvfile:
	fieldnames = ['     ', 'Politics', 'Film', 'Football', 'Business', 'Technology'];
	writer = csv.DictWriter(csvfile, fieldnames = fieldnames);
	writer.writeheader();
	for i in range(0,5):
		writer.writerow({'     ' : 'Cluster {}'.format(i+1), 'Politics' : polpercent[i], 'Film' :filmpercent[i], 'Football' : footpercent[i], 'Business' : buspercent[i], 'Technology' : techpercent[i]});        