
# !pip install pandas
# !pip install matplotlib
# !pip install scikit-learn
!where python

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import ast

movies = pd.read_csv('./data/tmdb_5000_movies.csv')
movies.shape

movies.head(2)

movies.info()

credit = pd.read_csv('./data/tmdb_5000_credits.csv')
credit.shape

credit.info()

"""### **Merging two dataframes**"""

tmdb = movies.merge(credit, on='title', how='left')
tmdb.shape

tmdb.info()

tmdb.isna().sum()

tmdb.head(2)

tmdb.columns

"""### **Cleaning and Transform data for Text Analysis**"""

tmdb = tmdb[['movie_id','title','overview','genres','keywords','cast','crew']]

print(tmdb.shape)
print(tmdb.columns)

tmdb.isna().sum()

mask = tmdb.isna().any(axis=1)
tmdb[mask]

tmdb.dropna(inplace=True)
tmdb.shape

tmdb.duplicated().sum()

"""* Transform the `string` name to the `list` structure"""

def get_names(text):
    lst = []
    for dictionary in ast.literal_eval(text):
        lst.append(dictionary['name'])
    return lst

tmdb['keywords'] = tmdb['keywords'].apply(get_names)
tmdb['genres'] = tmdb['genres'].apply(get_names)
tmdb['cast'] = tmdb['cast'].apply(get_names)

"""* Transform the `string` directories to the `list` structure"""

def get_directors(text):
    lst = []
    for dictionary in ast.literal_eval(text):
        if dictionary['job'].upper() == 'DIRECTOR':
            lst.append(dictionary['name'])
            break
    return lst

tmdb['crew'] = tmdb['crew'].apply(get_directors)

tmdb.head()

tmdb_tmp = tmdb.copy()

"""* Split the `text` overview into `list` structure"""

tmdb_tmp['overview'] = tmdb_tmp['overview'].apply(lambda x: x.split(' '))

"""* Removw `space` of proper nouns"""

def remove_space(lst):
    lst = [ item.replace(' ', '') for item in lst]
    return lst

tmdb_tmp['cast'] = tmdb_tmp['cast'].apply(remove_space)
tmdb_tmp['crew'] = tmdb_tmp['crew'].apply(remove_space)
tmdb_tmp['genres'] = tmdb_tmp['genres'].apply(remove_space)
tmdb_tmp['keywords'] = tmdb_tmp['keywords'].apply(remove_space)

"""* Combine several features to a `single one` tags"""

tmdb_tmp['tags'] = tmdb_tmp['overview'] + tmdb_tmp['genres'] + tmdb_tmp['keywords'] + tmdb_tmp['cast'] + tmdb_tmp['crew']

tmdb_tmp.head()

"""* Only select the movie_id, title and tags features"""

tmdb_tmp = tmdb_tmp[['movie_id','title','tags']]
tmdb_tmp.shape

"""* `Join` all the list to texts and apply `lower`"""

tmdb_tmp['tags'] = tmdb_tmp['tags'].apply(lambda x: ' '.join(x))

tmdb_tmp['tags'] = tmdb_tmp['tags'].apply(lambda x: x.lower())

tmdb_tmp.head()

"""* Transform the english words to a single form for each word form"""

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stems(text):
    lst = [ps.stem(word) for word in text.split(' ')]
    return ' '.join(lst)

tmdb_tmp['tags'] = tmdb_tmp['tags'].apply(stems)

tmdb = tmdb_tmp.copy()
tmdb.head()

"""### **Recommend appropriate films**"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

"""* *Counter vectorizer method*"""

cv = CountVectorizer(stop_words='english')
counter_matrix = cv.fit_transform(tmdb['tags']).toarray()

counter_matrix

counter_matrix.shape

"""* *TF - IDF matrix method*"""

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(tmdb['tags'])

pd.DataFrame(tfidf_matrix.toarray())

"""* *Applying consine similarity technique*"""

similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(similarities.shape)
similarities

similarities

"""* *Recommend function*"""

def recommend(title):
    index = tmdb[tmdb['title'] == title].index[0]
    distances = sorted(list(enumerate(similarities[index])), reverse=True, key=lambda x: x[1])
    for distance in distances[1:11]:
        print(tmdb.loc[distance[0], 'title'])

recommend('Batman')

index = tmdb[tmdb['title'] == 'Batman'].index[0]
distances = sorted(list(enumerate(similarities[index])), reverse=True, key=lambda x: x[1])

# visualize the similarity among movies
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax1 = sns.lineplot([distance[1] for distance in distances[0:11]], ax=ax[0])
ax1.set(xlabel='Similar movies')

ax2 = sns.histplot([distance[1] for distance in distances[0:11]], kde=True, ax=ax[1])
ax2.set(xlabel='Similarity levels')

"""### **Dump to files for using again**"""

import pickle

# pickle.dump(tmdb, open('artificats/movies.pkl', 'wb'))
# pickle.dump(similarities, open('artificats/similarities.pkl', 'wb'))

import requests
import json

url = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzZmVkMTgxYWZiZTI4NDc2OWM2YTQ5NTMzNGRjNjZlYSIsInN1YiI6IjY1ZDIxYzcxNmVlY2VlMDE4YTM5MmZkNyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.H8tZ1WzpXyiA4P_sOGOalbL7Th9DtQhixCgpTix93qM"
}

response = requests.get(url, headers=headers)

json.loads(response.text)