from scipy.sparse import csr_matrix
import re
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

n_components = 23
chunk_size = 1000
cosine_sim_pca_list = []

movies_df = pd.read_csv('C:/Users/werak/Downloads/ml-25m/ml-25m/movies.csv')
tag_df = pd.read_csv('C:/Users/werak/Downloads/ml-25m/ml-25m/tags.csv')

def preprocess_tag(tag):
    tag = re.sub('[^a-zA-Z]', ' ', tag)
    tag = tag.lower()
    tag = [lemmatizer.lemmatize(word) for word in tag.split()]
    return ' '.join(tag)

tag_df['tag'] = tag_df['tag'].fillna('').apply(preprocess_tag)
tag_df = tag_df.groupby('movieId')['tag'].apply(' '.join).reset_index()
movies_df = movies_df.merge(tag_df, on='movieId', how='left')

movies_df['genres_tag'] = movies_df['genres'] + ' ' + movies_df['tag'].fillna('')

def extract_release_year(title):
    pattern = r"\((\d{4})\)$"
    match = re.search(pattern, title)
    if match:
        return match.group(1)
    else:
        return None

movies_df["release_year"] = movies_df["title"].apply(extract_release_year)

tfidf = TfidfVectorizer(stop_words='english')
movies_df['genres_tag'] = movies_df['genres_tag'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies_df['genres_tag'])

pca = IncrementalPCA(n_components=n_components, batch_size=chunk_size)

pca = IncrementalPCA(n_components=n_components, batch_size=chunk_size)

for i in range(0, tfidf_matrix.shape[0], chunk_size):
    chunk = tfidf_matrix[i:i+chunk_size]
    chunk_dense = chunk.toarray()
    pca.partial_fit(chunk_dense)
    chunk_pca = pca.transform(chunk_dense)

    cosine_sim_pca_list.append(chunk_pca)

cosine_sim_pca = np.concatenate(cosine_sim_pca_list, axis=0)

n_neighbors = 6
knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine')
knn.fit(cosine_sim_pca)

def get_recommendations(title, knn=knn, df=movies_df, top_n=10):
    matching_movies = df[df['title'].str.contains(title, case=False, na=False)]
    if not matching_movies.empty:
        idx = matching_movies.index[0]
        distances, indices = knn.kneighbors(cosine_sim_pca[idx].reshape(1, -1))
        movie_indices = indices.flatten()[1:top_n+1]
        return df['title'].iloc[movie_indices].tolist()
    else:
        return f"No movie found with the title '{title}'."

recommendations = get_recommendations('Iron Man')
print(recommendations)
