import math
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

from ..database.database import *

load_dotenv()

MAX_FACET_COUNT = int(os.getenv("MAX_FACET_COUNT"))
MIN_FACET_COUNT = int(os.getenv("MIN_FACET_COUNT"))

ACCURACY_DIVERSITY_TRADEOFF = float(os.getenv("ACCURACY_DIVERSITY_TRADEOFF"))

ADDED_TREND_CONTRIBUTION = float(os.getenv("ADDED_TREND_CONTRIBUTION"))
LIKED_TREND_CONTRIBUTION = float(os.getenv("LIKED_TREND_CONTRIBUTION"))
MODEL_TREND_CONTRIBUTION = float(os.getenv("MODEL_TREND_CONTRIBUTION"))
REMOVED_TREND_CONTRIBUTION = float(os.getenv("REMOVED_TREND_CONTRIBUTION"))
REJECTED_TREND_CONTRIBUTION = float(os.getenv("REJECTED_TREND_CONTRIBUTION"))

ATTR_SCALES = [[0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,60], [0,220], [0,11], [0,1]]
FACET_LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
KL_DIVERGENCE = tf.keras.losses.KLDivergence()

###################################
######### DATA STRUCTURES #########
###################################

class WeightedKMeans(KMeans):

    def __init__(self, n_clusters=8, weights=None, **kwargs):
        self.weights = weights
        super().__init__(n_clusters=n_clusters, **kwargs)

    def _euclidean_distance(self, a, b):
        if self.weights is None:
            return np.linalg.norm(a - b)
        else:
            return np.sqrt(np.sum(self.weights * (a - b) ** 2, axis=-1))

##########################################
######### FACET MODEL OPERATIONS #########
##########################################

# Build initial facet distribution model
def buildModel(track_audio_data):

    # Prepare data
    track_data = extractDataPoints(track_audio_data)
    track_data = normalize_data(track_data)

    # Calculate attribute weights
    attr_weights = genAttributeWeights(track_data)

    # Determine facet clusters
    facet_clusters, n_facets = findOptimalKmeansClusters(track_data, attr_weights)

    # Determine similarity threshold for each model track
    sim_thresholds = getSimilarityThresholds(track_data, attr_weights, n_facets)

    # facet_clusters, sim_thresholds = nn1(track_data, attr_weights, n_facets)

    # Assign model track classifications
    for i, cluster in enumerate(facet_clusters):
        for track_index in cluster:
            track_audio_data[track_index]["facet"] = FACET_LABELS[i]

    # Assign model track similarity threshold
    for i, track in enumerate(track_audio_data):
        track['sim_threshold'] = sim_thresholds[i]

    return track_audio_data, attr_weights

# Calculate attribute weights from attribute variances
def genAttributeWeights(model_dps):

    print("Calculating attribute weights...")

    # Extract attribute values from data points
    attribute_values = np.array(model_dps)

    # Calculate the variance of each attribute
    attribute_variances = np.var(attribute_values, axis=0)

    # Calculate attribute weights based on inverse variance and normalize
    attribute_weights = 1 / attribute_variances
    attribute_weights = attribute_weights / np.sum(attribute_weights) * len(attribute_weights)


    print("Attribute weights generated successfully!")
    print("Attribute weights:", attribute_weights)

    return attribute_weights

# Find optimal facet model cluster distribution based on silhouette score
def findOptimalKmeansClusters(data, attr_weights):

    print("Building facet cluster model...")

    best_silhouette_score = -1
    best_k = 0
    best_cluster_assignment = None

    for k in range(MIN_FACET_COUNT, MAX_FACET_COUNT + 1):
        # Apply KMeans clustering
        kmeans = WeightedKMeans(n_clusters=k, weights=attr_weights, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_k = k
            best_cluster_assignment = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                best_cluster_assignment[label].append(i)  # Include the ID in the cluster assignment
    
    print("Optimal facet distribution found!")
    print("Optimal number of clusters: ", best_k)
    
    facet_classification = [best_cluster_assignment[i] for i in range(best_k)]
    
    return facet_classification, best_k

# Find optimal facet model cluster distribution using nn-1
def nn1(model_data, attr_weights, n_facets):

    def weighted_distance(x, y):
            return np.sqrt(np.sum(attr_weights * (x - y) ** 2))

    model_data = np.array(model_data)
    k =  math.ceil(len(model_data) / n_facets)

    knn_model = NearestNeighbors(n_neighbors = k + 1, algorithm='brute', metric=weighted_distance)
    knn_model.fit(model_data)
    distances, k_nearest_indices = knn_model.kneighbors(model_data)

    candidate_facet_clusters = [set(index for index, neighbors in enumerate(k_nearest_indices) if i in neighbors) for i in range(len(k_nearest_indices))]

    final_clusters = []
    candidate_facet_clusters.sort(key=len, reverse=True)
    for cluster in candidate_facet_clusters:
        if not any(cluster.issubset(existing_cluster) for existing_cluster in final_clusters):
            final_clusters.append(cluster)

    similarity_thresholds = distances[:, -1]

    print("Optimal facet distribution found!")
    print("Optimal number of clusters: ", len(final_clusters))

    return final_clusters, similarity_thresholds

# Calculate similarity thresholds inside the model
def getSimilarityThresholds(model_data, attr_weights, n_facets):
    
    def weighted_distance(x, y):
            return np.sqrt(np.sum(attr_weights * (x - y) ** 2))
        
    # Extract features from model tracks
    features = np.array(model_data)
    
    # Fit KNN model
    k =  math.ceil(len(model_data) / n_facets)
    knn_model = NearestNeighbors(n_neighbors = k + 1, algorithm='brute', metric=weighted_distance)
    knn_model.fit(features)
    
    # Calculate distances to k-nearest neighbors for each data point
    distances, _ = knn_model.kneighbors(features)
    
    similarity_thresholds = distances[:, -1] 
    # similarity_thresholds = np.mean(distances[:, 1:], axis=1) 
    
    return similarity_thresholds

# Prepare model data for Post-Ranking
def prepareModelData(model_data, newly_added_data, newly_liked_data, removed_tracks, rejected_tracks, stream_count, last_stream_ts, n_facets, attr_weights, current_ts, last_cycle_ts):

    new_model_data = model_data + newly_added_data + newly_liked_data

    # Extract data points and normalize them
    new_model_dps = normalize_data(extractDataPoints(new_model_data))

    # Get similarity thresholds and assign them to each track_data
    sim_thresholds = getSimilarityThresholds(new_model_dps, attr_weights, n_facets)
    for i, track_data in enumerate(new_model_data):
        track_data['sim_threshold'] = sim_thresholds[i]

    # Extract relevant tracks and calculate their attributes
    relevant_model_tracks = [track_data for track_data in model_data if track_data['id'] in stream_count]
    relevant_tracks = relevant_model_tracks + newly_added_data + newly_liked_data + removed_tracks + rejected_tracks
    neighborhood_dists, classifications = getNeighborhoodDistributions(relevant_tracks, model_data, n_facets, attr_weights, True)

    for i, track_data in enumerate(relevant_tracks):
        default_ts = last_cycle_ts

        if track_data in newly_added_data or track_data in newly_liked_data:
            track_data['facet'] = classifications[i]

        if track_data in newly_added_data:
            default_ts = track_data['added_at']

        track_data['stream_count'] = stream_count.get(track_data['id'], 0)
        track_data['recency_weight'] = (current_ts - last_stream_ts.get(track_data['id'], default_ts)) / (24 * 60 * 60)
        track_data['nn_dist'] = neighborhood_dists[i]

    # Split relevant tracks into model, newly added, newly liked, and removed tracks
    split_index_model = len(relevant_model_tracks)
    split_index_added = split_index_model + len(newly_added_data)
    split_index_liked = split_index_added + len(newly_liked_data)
    split_index_removed = split_index_liked + len(removed_tracks)

    relevant_model_tracks = relevant_tracks[:split_index_model]
    newly_added_data = relevant_tracks[split_index_model:split_index_added]
    newly_liked_data = relevant_tracks[split_index_added:split_index_liked]
    removed_tracks = relevant_tracks[split_index_liked:split_index_removed]
    rejected_tracks = relevant_tracks[split_index_removed:]

    model_data = new_model_data[:len(model_data)] + newly_added_data + newly_liked_data

    return [model_data, relevant_model_tracks, newly_added_data, newly_liked_data, removed_tracks, rejected_tracks]

# Extract initial facet model distribution
def extractInitPrefDist(model_data):

    print("Extracting initial facet model distribution...")

    preference_dist = {}

    # Calculate facet counts
    for track in model_data:
        facet = track['facet']
        preference_dist[facet] = preference_dist.get(facet, 0) + 1

    # Normalize counts to get facet relative frequencies
    total_tracks = len(model_data)
    preference_dist = {facet: count / total_tracks for facet, count in preference_dist.items()}
    preference_dist = [preference_dist[facet] for facet in sorted(preference_dist)]

    print("Facet Preference Distribution:", preference_dist)

    return preference_dist

# Extract recent facet preference
def extractRecentTrendDist(rel_model_data, newly_added_data, newly_liked_data, removed_data, rejected_data, n_facets):

    print("Extracting recent facet trend distribution...")

    # Initialize recent trend distribution
    recent_trend_dist = np.zeros(n_facets)

    # Extract stream weights for added, liked, and model data and perform min-max scaling
    added_stream_weights = minMaxScaling([track['stream_count'] for track in newly_added_data])
    liked_stream_weights = minMaxScaling([track['stream_count'] for track in newly_liked_data])
    model_stream_weights = minMaxScaling([track['stream_count'] for track in rel_model_data])

    # Compute newly added data
    for i, track in enumerate(newly_added_data):
        recency_weight = (ADDED_TREND_CONTRIBUTION + (1 - ADDED_TREND_CONTRIBUTION) * added_stream_weights[i]) ** track['recency_weight']
        recent_trend_dist += recency_weight * np.array(track['nn_dist'])

    # Compute newly liked data
    for i, track in enumerate(newly_liked_data):
        recency_weight = (LIKED_TREND_CONTRIBUTION + (ADDED_TREND_CONTRIBUTION - LIKED_TREND_CONTRIBUTION) * liked_stream_weights[i]) **  track['recency_weight']
        recent_trend_dist += recency_weight * np.array(track['nn_dist'])

    # Compute relevant model data
    for i, track in enumerate(rel_model_data):
        recency_weight = (MODEL_TREND_CONTRIBUTION + (LIKED_TREND_CONTRIBUTION - MODEL_TREND_CONTRIBUTION) * model_stream_weights[i]) ** track['recency_weight']
        recent_trend_dist += recency_weight * np.array(track['nn_dist'])

    # Compute removed data
    for track in removed_data:
        recent_trend_dist += REMOVED_TREND_CONTRIBUTION * np.array(track['nn_dist'])

    # Compute removed data
    for track in rejected_data:
        recent_trend_dist += REJECTED_TREND_CONTRIBUTION * np.array(track['nn_dist'])

    # Normalize recent trend distribution and convert to list
    recent_trend_dist[recent_trend_dist < 0] = 0
    recent_trend_dist = recent_trend_dist / sum(recent_trend_dist)

    print("Recent facet trend distribution successfully extracted!")
    print("Recent trend distribution:", recent_trend_dist)

    return recent_trend_dist.tolist()

###########################################
######### POST-RANKING OPERATIONS #########
###########################################

# Select final recommendations from pool
def postRanking(rec_data, goal_dist, n_select, model_data, attr_weights, rec_protocol):

    print("Initiating post-ranking process...")

    # Generate recommendations' baseline ratings and neighborhood facet distributions
    baseline_ratings = genBaselineRating(rec_data, model_data, attr_weights)
    neighborhood_dists = getNeighborhoodDistributions(rec_data, model_data, len(goal_dist), attr_weights)

    if rec_protocol == "TASTES":

        # Initialize list of top recommendations
        top_recs_index = []
        top_recs_size = 0
        top_recs_rating = 0
        top_recs_facet_dists = []

        #Add track with highest inclusive value from the initial pool until desired number of final recommendations
        while top_recs_size < n_select:
            most_valuable_rec = 0
            highest_value = -math.inf
            kl_divs = genKLDivergences(goal_dist, top_recs_facet_dists, neighborhood_dists, top_recs_index)
            inclusion_ratings = genInclusionRatings(top_recs_rating, baseline_ratings, top_recs_index)

            for i, rec in enumerate(rec_data):

                if i in top_recs_index:
                    continue

                inclusion_value = (1 - ACCURACY_DIVERSITY_TRADEOFF) * inclusion_ratings[i] + ACCURACY_DIVERSITY_TRADEOFF * kl_divs[i]

                if inclusion_value > highest_value:
                    most_valuable_rec = i
                    highest_value = inclusion_value

            top_recs_index.append(most_valuable_rec)
            top_recs_size += 1
            top_recs_rating += baseline_ratings[most_valuable_rec]
            top_recs_facet_dists.append(neighborhood_dists[most_valuable_rec])

        top_recs = [rec_data[i] for i in top_recs_index]
        final_relevance_ratings = [baseline_ratings[i] for i in top_recs_index]

    else:
        top_recs = rec_data[:]
        top_recs_facet_dists = neighborhood_dists[:]
        final_relevance_ratings = baseline_ratings[:]
        top_recs_rating = sum(final_relevance_ratings)


    final_facet_dist = [sum(x) / n_select for x in zip(*top_recs_facet_dists)]
    rec_tensor = tf.convert_to_tensor(final_facet_dist, dtype=tf.float32)
    goal_tensor = tf.convert_to_tensor(goal_dist, dtype=tf.float32)
    final_kl_div = KL_DIVERGENCE(goal_tensor, rec_tensor).numpy()

    print("Post ranking process completed!")
    print("Recommendations selected:", [rec['id'] for rec in top_recs])
    print("Final baseline ratings:", final_relevance_ratings, "{Total Relevance:", top_recs_rating, "}")
    print("Final facet distribution:", final_facet_dist, "{KL Divergence:", final_kl_div, "}")
    
    return top_recs, float(final_kl_div), top_recs_rating

# Generate baseline ratings for recommendations
def genBaselineRating(recs_data, model_data, attr_weights):

    print("Generating recommendations' baseline ratings...")

    # Extract data points and normalize all data
    rec_data_points = extractDataPoints(recs_data)
    model_data_points = extractDataPoints(model_data)
    normalized_data = normalize_data(rec_data_points + model_data_points)
    split_index = len(rec_data_points)
    rec_data_points = normalized_data[:split_index]
    model_data_points = normalized_data[split_index:]

    # Extract model similarity thresholds and relevances
    similarity_thresholds = [track['sim_threshold'] for track in model_data]
    model_relevances = [track['relevance'] for track in model_data]

    results = []

    for rec in rec_data_points:
        rating = 0

        for i, model_track in enumerate(model_data_points):
            # Calculate similarity between tracks
            squared_diff = (np.array(rec) - np.array(model_track)) ** 2
            squared_diff = squared_diff * attr_weights
            similarity = np.sqrt(np.sum(squared_diff))

            # Check if similarity is below threshold and calculate rating
            if similarity < similarity_thresholds[i]:
            #    rating += (similarity_thresholds[i] / similarity ) * model_relevances[i]
                rating += (similarity_thresholds[i] - similarity) / similarity_thresholds[i] * model_relevances[i]

        results.append(rating)

    # Normalize the results
    results = np.array(results)
    normalized_results = (results - np.min(results)) / (np.max(results) - np.min(results))

    print("Baseline ratings generated successfully!")
    print("Baseline Ratings: ", normalized_results.tolist())

    return normalized_results.tolist()

# Calculate neighborhood distributions in the facet model
def getNeighborhoodDistributions(track_data, model_data, n_facets, attr_weights, classify = False):
    
    print("Generating recommendations' neighborhood distributions...")

    # Extract data points and normalize all data
    track_data_points = extractDataPoints(track_data)
    model_data_points = extractDataPoints(model_data)
    normalized_data = normalize_data(track_data_points + model_data_points)
    split_index = len(track_data_points)
    track_data_points = normalized_data[:split_index]
    model_data_points = normalized_data[split_index:]
    model_data_points = np.array(model_data_points)
    track_data_points = np.array(track_data_points)

    # Extract model ids and classifications
    model_ids = [track['id'] for track in model_data]
    model_classifications = [track['facet'] for track in model_data]

    def weighted_distance(x, y):
        return np.sqrt(np.sum(attr_weights * (x - y) ** 2))

    k =  math.ceil(len(model_data_points) / n_facets)
    nn_model = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=weighted_distance)
    nn_model.fit(model_data_points)
        
    neighborhood_distributions = []

    for i, track_dp in enumerate(track_data_points):

        # Find indices of K nearest neighbors of track, excluding the track
        if track_data[i]['id'] in model_ids:
            _, indices = nn_model.kneighbors(track_dp.reshape(1, -1), n_neighbors=k + 1)
            indices = [indices[0][1:]]

        else:
            _, indices = nn_model.kneighbors(track_dp.reshape(1, -1), n_neighbors=k)

        # Initialize neighborhood distribution
        neighborhood_distribution = [0] * n_facets

        # Update neighborhood distribution based on neighbors' facets
        for idx in indices[0]:
            facet_index = FACET_LABELS.index(model_classifications[idx])  # Get facet index of neighbor
            neighborhood_distribution[facet_index] += 1

        # Normalize neighborhood distribution
        neighborhood_distribution = [dist / k for dist in neighborhood_distribution]

        neighborhood_distributions.append(neighborhood_distribution)

    if classify:
        classifications = []

        for dist in neighborhood_distributions:
            max_value_index = dist.index(max(dist))
            classifications.append(FACET_LABELS[max_value_index])

        return neighborhood_distributions, classifications
    
    print("Neighborhood distributions generated successfully!")
    print("Neighborhood Distributions: ", str(neighborhood_distributions))

    return neighborhood_distributions

# Calculate KL_divergence for each track addition to final recommendations
def genKLDivergences(goal_dist, current_rec_dist, neighboorhood_dists, picked_recs):

    result = []
    goal_tensor = tf.convert_to_tensor(goal_dist, dtype=tf.float32)
    min_div_value = math.inf
    max_div_value = -math.inf
    
    for i, dist in enumerate(neighboorhood_dists):            
        
        if i in picked_recs:
            result.append(-math.inf)
            continue
                
        added_dist = np.array(current_rec_dist + [dist])
        added_dist = np.mean(added_dist, axis=0)
        
        added_tensor = tf.convert_to_tensor(added_dist, dtype=tf.float32)

        kl_divergence = KL_DIVERGENCE(goal_tensor, added_tensor).numpy()

        if kl_divergence < min_div_value:
            min_div_value = kl_divergence
        
        if kl_divergence > max_div_value:
            max_div_value = kl_divergence
        
        result.append(kl_divergence)
    
    result = minMaxScaling(result, scale = [min_div_value, max_div_value])

    return result

def genInclusionRatings(current_recs_rating, baseline_ratings, picked_recs):

    result = []
    min_value = math.inf
    max_value = -math.inf

    for i, rating in enumerate(baseline_ratings):

        if i in picked_recs:
            result.append(-math.inf)
            continue
        
        inclusion_rating = (current_recs_rating + rating) / (len(picked_recs) + 1)

        if inclusion_rating < min_value:
            min_value = inclusion_rating
        
        if inclusion_rating > max_value:
            max_value = inclusion_rating

        result.append(inclusion_rating)

    result = minMaxScaling(result, scale = [min_value, max_value])

    return result

#################################
######### DATA HANDLING #########
#################################

# Extract data points from track audio data
def extractDataPoints(audio_data):

    results = []
    key_sum = 0
    key_count = 0

    for track in audio_data:
        key = track["key"]
        if key != -1:
            key_sum += key
            key_count += 1

    avg_key = key_sum / key_count if key_count > 0 else -1

    for track in audio_data:
        key = avg_key if track["key"] == -1 else track["key"]
        
        results.append([
            track["acousticness"],
            track["danceability"],
            track["energy"],
            track["instrumentalness"],
            track["liveness"],
            track["speechiness"],
            track["valence"],
            abs(track["loudness"]),  # Ensure loudness is positive
            track["tempo"],
            key,
            track["mode"]
        ])

    return results

# Normalize data points
def normalize_data(data):
    
    # Check if all data points have the same number of attributes
    num_attributes = len(data[0])  # Excluding the track ID
    if any(len(point) != num_attributes for point in data):
        raise ValueError("Not all data points have the same number of attributes")

    # Extract attribute values from data points
    attribute_values = np.array(data)

    # Transpose data to operate on columns (attributes) instead of rows (tracks)
    attribute_values_transposed = attribute_values.T

    # Normalize each attribute separately
    normalized_attribute_values = [minMaxScaling(attribute_values_transposed[i], ATTR_SCALES[i]) for i in range(num_attributes)]

    # Combine track IDs with normalized attribute values
    normalized_data = [list(attr_values) for attr_values in zip(*normalized_attribute_values)]

    return normalized_data

# Normalize data (to scale if given)
def minMaxScaling(values, scale = None):

    if len(values) == 0:
        return values
    
    elif scale is not None and scale != [0,0]:
        min_val = scale[0]
        max_val = scale[-1]
    else:
        min_val = min(values)
        max_val = max(values)

        if not (max_val - min_val) > 0:
            return [1] * len(values)

    scaled_values = [(x - min_val) / (max_val - min_val) for x in values]

    return scaled_values

# Perform principle component analysis on data
def apply_pca(track_data_normalized):

    track_ids = [point[0] for point in track_data_normalized]
    attribute_values = np.array([point[1:] for point in track_data_normalized])

    pca = PCA()
    track_data_pca = pca.fit_transform(attribute_values)

    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance_ratio_cumsum >= EXPLAINED_VARIANCE_THRESHOLD) + 1

    pca = PCA(n_components=n_components)
    track_data_pca = pca.fit_transform(attribute_values)

    print("PCA components: " + str(n_components))

    track_data_pca_final = np.column_stack([track_ids, track_data_pca])

    return track_data_pca_final.tolist()