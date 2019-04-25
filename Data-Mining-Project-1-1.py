import pandas as pd
from wordcloud import WordCloud as wc
from wordcloud import ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS
import numpy as np
from PIL import Image
from os import path

#open train_set.csv
df = pd.read_csv('train_set.csv', sep = '\t');

#take the current path
d = path.dirname(__file__)

#open the mask that is going to be used for the wordcloud
stormtrooper_mask = np.array(Image.open(path.join(d, "stormtrooper_mask.png")))

#get some ready stopwords and add some of our choice
stopwords = set(STOPWORDS);
stopwords.add("will");
stopwords.add("well");
stopwords.add("one");
stopwords.add("even");
stopwords.add("now");
stopwords.add("many");
stopwords.add("still");
stopwords.add("since");
stopwords.add("also");
stopwords.add("said");
stopwords.add("say");
stopwords.add("us");
stopwords.add("much");
stopwords.add("got");
stopwords.add("want");
stopwords.add("going");
stopwords.add("thing");
stopwords.add("rather");
stopwords.add("although");
stopwords.add("anything");
stopwords.add("every");
stopwords.add("already");
stopwords.add("something");
stopwords.add("says");
stopwords.add("least");
stopwords.add("take");
stopwords.add("put");
stopwords.add("go");
stopwords.add("whether");
stopwords.add("may");
stopwords.add("look");
stopwords.add("saying");
stopwords.add("use");
stopwords.add("whose");
stopwords.add("almost");
stopwords.add("really");
stopwords.add("whithin");
stopwords.add("other");
stopwords.add("re");
stopwords.add("see");
stopwords.add("way");


#taking the data that refers to politics
data = df.ix[df['Category'] == 'Politics'];
print ("Politics");
#taking the content that refers to politics
politics = data['Content'];
datstring = ''.join(politics);
#generating the wordcloud with max words projected equal to max_words
wordcloud = wc(background_color="white", mask=stormtrooper_mask , max_words=200, stopwords=stopwords, random_state=42).generate(datstring);
#printing the png wordcloud for politics
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#THE EXACT SAME PROCESS IS DONE FOR EVERY CATEGORY BELOW

data = df.ix[df['Category'] == 'Business'];
print ("Business");
business = data['Content'];
datstring = ''.join(business);
wordcloud = wc(background_color="white", mask=stormtrooper_mask , max_words=200, stopwords=stopwords, random_state=42).generate(datstring);
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

data = df.ix[df['Category'] == 'Film'];
print ("Film");
film = data['Content'];
datstring = ''.join(film);
wordcloud = wc(background_color="white", mask=stormtrooper_mask , max_words=200, stopwords=stopwords, random_state=42).generate(datstring);
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

data = df.ix[df['Category'] == 'Football'];
print ("Football");
football = data['Content'];
datstring = ''.join(football);
wordcloud = wc(background_color="white", mask=stormtrooper_mask , max_words=200, stopwords=stopwords, random_state=42).generate(datstring);
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

data = df.ix[df['Category'] == 'Technology'];
print ("Technology");
technology = data['Content'];
datstring = ''.join(technology);
wordcloud = wc(background_color="white", mask=stormtrooper_mask , max_words=200, stopwords=stopwords, random_state=42).generate(datstring);
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
