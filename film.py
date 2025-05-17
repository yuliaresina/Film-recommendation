import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 


def load_data(drop_stopwords=False, min_tags=3):
    # Load the data
    movies = pd.read_csv("movies.csv", index_col=0)
    tags = pd.read_csv("tags.csv", index_col=0)

    # Clean up genres
    movies["genres"] = movies["genres"].fillna("").str.split("|")
    movies["genre_str"] = movies["genres"].apply(lambda x: ",".join(x))
    genre_dummies = movies["genre_str"].str.get_dummies(sep=",")
    movies = pd.concat([movies, genre_dummies], axis=1)
    movies.drop(columns=["genre_str"], inplace=True)

    # Clean up tags
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    if drop_stopwords:
        tags = tags[~tags["tag"].isin(ENGLISH_STOP_WORDS)]
    tags = tags[tags["tag"].str.len() > 2]

    # Combine tags per movie
    combined_tags = tags.groupby("movieId")["tag"].apply(" ".join).reset_index()
    combined_tags["tag_count"] = combined_tags["tag"].apply(lambda x: len(x.split()))
    filtered_tags = combined_tags[combined_tags["tag_count"] >= min_tags]

    # Merge with movies
    movies.drop_duplicates(subset="title", keep="first", inplace=True)
    movies = movies.merge(filtered_tags[["movieId", "tag"]], on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    
    # Clean up titles
    movies["title"] = movies["title"].str.strip()

    # Return the cleaned DataFrame
    return movies



class MoviePreprocessor:
    def __init__(self, min_tags=3, drop_stopwords=False, drop_duplicates=True):
        self.min_tags = min_tags
        self.drop_stopwords = drop_stopwords
        self.drop_duplicates = drop_duplicates
        self.genre_columns = []

    def _clean_genres(self, movies):
        movies["genres"] = movies["genres"].fillna("").str.split("|")
        movies = movies[movies["genres"].apply(len) > 0]
        movies["genre_str"] = movies["genres"].apply(lambda x: ",".join(x))
        genre_dummies = movies["genre_str"].str.get_dummies(sep=",")
        self.genre_columns = genre_dummies.columns.tolist()
        movies = pd.concat([movies, genre_dummies], axis=1)
        return movies.drop(columns=["genre_str"])

    def _clean_tags(self, tags):
        tags.dropna(subset=["tag"], inplace=True)
        tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
        if self.drop_stopwords:
            tags = tags[~tags["tag"].isin(ENGLISH_STOP_WORDS)]
        tags = tags[tags["tag"].str.len() > 2]
        return tags

    def _combine_tags(self, tags):
        tag_data = tags.groupby("movieId")["tag"].apply(" ".join).reset_index()
        tag_data["tag_count"] = tag_data["tag"].apply(lambda x: len(x.split()))
        return tag_data[tag_data["tag_count"] >= self.min_tags][["movieId", "tag"]]

    def fit_transform(self, movies_path, tags_path):
        movies = pd.read_csv(movies_path, index_col=0)
        tags = pd.read_csv(tags_path, index_col=0)

        if self.drop_duplicates:
                movies.drop_duplicates(subset="title", inplace=True)

        movies = self._clean_genres(movies)
        tags = self._clean_tags(tags)
        combined_tags = self._combine_tags(tags)

        movies = movies.merge(combined_tags, on="movieId", how="left")
        movies["tag"] = movies["tag"].fillna("")
        movies = movies[(movies["tag"].str.strip() != "") & (movies["genres"].apply(len) > 0)]

        return movies

# 2. Compute similarity 

def compute_similarity(movies, scale_type="minmax"):
    # 1. TF-IDF on tags
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10_000)
    tfidf_matrix = vectorizer.fit_transform(movies["tag"]).toarray()  # dense array

    # 2. Genre features
    genre_cols = movies.columns.difference(["title", "genres", "tag", "movieId"])
    genre_matrix = movies[genre_cols].fillna(0).values.astype(np.float32)
    genre_matrix *= 2  # Optional: boost genre weight

    # 3. Combine
    combined_matrix = np.hstack((genre_matrix, tfidf_matrix))

    # 4. Scale the combined feature matrix
    if scale_type == "minmax":
        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    combined_scaled = scaler.fit_transform(combined_matrix)

    # 5. Cosine similarity
    return cosine_similarity(combined_scaled)

# 3. Recommend similar movies
def recommend_movies(movie_title, movies, cosine_sim, num_recommendations=5):
    # Case-insensitive title match
    matched_titles = movies[movies["title"].str.lower() == movie_title.lower()]
    if matched_titles.empty:
        return f"Movie '{movie_title}' is not in a database."

    # Get the first matching index (assuming no duplicate titles)
    idx = matched_titles.index[0]
    movie_genres = set(movies.iloc[idx]["genres"])

    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]

    # Filter by genre overlap
    filtered_recommendations = []
    for movie_idx, score in sim_scores:
        recommended_genres = set(movies.iloc[movie_idx]["genres"])
        if movie_genres & recommended_genres:
            filtered_recommendations.append((movie_idx, score))
        if len(filtered_recommendations) == num_recommendations:
            break

    # Check if we found any recommendations
    if not filtered_recommendations:
        return f"No similar movies found for '{movie_title}'."

    # Build the recommendations DataFrame
    recommendations = movies.iloc[[i[0] for i in filtered_recommendations]][["title", "genres", "tag"]].copy()
    recommendations["similarity_score"] = [i[1] for i in filtered_recommendations]
    recommendations.reset_index(drop=True, inplace=True)
    
    return recommendations


### with help of classmate and chatgpt