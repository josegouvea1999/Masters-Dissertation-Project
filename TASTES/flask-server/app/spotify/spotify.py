import os
import random
import spotipy
import base64
import time
import math

from app import app
from flask import jsonify
from flask_cors import cross_origin
from dotenv import load_dotenv
from ..database.database import *
from ..auth.auth import getCurrentUserToken, getUserId
from datetime import datetime
from ..tastes.tastes import buildModel, extractInitPrefDist, extractRecentTrendDist, postRanking, prepareModelData
from collections import Counter, OrderedDict

load_dotenv()

RECOMMENDATION_PROTOCOL = os.getenv('RECOMMENDATION_PROTOCOL')
PREF_DIST_UPDATE_FACTOR = int(os.getenv('PREF_DIST_UPDATE_FACTOR'))

RADIO_SIZE = int(os.getenv("RADIO_SIZE"))
RADIO_SUFFIX = os.getenv("RADIO_SUFFIX")

MAX_STREAM_REQUESTS = int(os.getenv("MAX_STREAM_REQUESTS"))
MAX_REC_REQUESTS = int(os.getenv("MAX_REC_REQUESTS"))
REC_REQ_SIZE = int(os.getenv("REC_REQ_SIZE"))
POOL_SIZE_FACTOR = int(os.getenv("POOL_SIZE_FACTOR"))

RELEVANCE_FUNCTION_SLOPE = int(os.getenv("RELEVANCE_FUNCTION_SLOPE"))
RELEVANCE_FUNCTION_MIDPOINT = float(os.getenv("RELEVANCE_FUNCTION_MIDPOINT"))

PLAYLIST_DECAY_FACTOR = float(os.getenv("PLAYLIST_DECAY_FACTOR"))
MODEL_DECAY_FACTOR = float(os.getenv("MODEL_DECAY_FACTOR"))
STREAM_COUNT_FACTOR = float(os.getenv("STREAM_COUNT_FACTOR"))
SEED_SUCCESS_FACTOR = float(os.getenv("SEED_SUCCESS_FACTOR"))
ADDED_RELEVANCE_CUTOFF = float(os.getenv("ADDED_RELEVANCE_CUTOFF"))
SUCCESSFUL_RELEVANCE_CUTOFF = float(os.getenv("SUCCESSFUL_RELEVANCE_CUTOFF"))
LIKED_RELEVANCE_CUTOFF = float(os.getenv("LIKED_RELEVANCE_CUTOFF"))
UNSUCCESSFUL_RELEVANCE_CUTOFF = float(os.getenv("UNSUCCESSFUL_RELEVANCE_CUTOFF"))

@app.route('/refresh/<playlist_id>')
@cross_origin()
def refresh(playlist_id):
          
    try:
        token = getCurrentUserToken()
        sp = spotipy.Spotify(auth=token.access_token)
        playlist = sp.playlist(playlist_id)

        if playlist['tracks']['total'] < 20:
            return TypeError("Playlist selected must have at least 20 tracks")
        
        
        #Check if user is registered in database
        user_id = getUserId()
        if not checkUserSchema(user_id):
            initializeTables(user_id)
            
        radio_id = getPlaylistRadio(user_id, playlist_id)
        radio_name = playlist['name'] + RADIO_SUFFIX
        
        if radio_id is None:

            print("Generating ", radio_name, " playlist with ", RECOMMENDATION_PROTOCOL, " recommendation protocol...")

            # Extract initial playlist ids and relevances
            track_relevances, model_data = genTracklistData(playlist)
            for track in model_data:
                track['relevance'] = track_relevances[track['id']]
            
            # Generate initial recommendation pool
            final_recs = getRecommendations(track_relevances, RECOMMENDATION_PROTOCOL)            
            
            # FOR TESTING PURPUSES:
            # publishPlaylistRadio(user_id, "Recommendation Pool" , [track['id'] for track in final_recs])

            # FOR TESTING PURPUSES:
            # rec_playlist = sp.playlist("7xn1KH7xRDvwoje7zIP4ff")
            # _, final_recs = genTracklistData(rec_playlist)
            # for rec in final_recs:
            #     rec['seed_id'] = "NS"

            # Generate initial model data
            model_data, attr_weights = buildModel(model_data)
            facet_pref_dist = extractInitPrefDist(model_data)
            n_facets = len(facet_pref_dist)

            # Perform post_ranking
            final_recs, kl_divergence, total_relevance = postRanking(final_recs, facet_pref_dist, RADIO_SIZE, model_data, attr_weights, RECOMMENDATION_PROTOCOL)

            # FOR TESTING PURPUSES:
            # for rel_div_factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
            #    final_rec_ids = postRanking(final_recs, facet_pref_dist, RADIO_SIZE, model_data, attr_weights, l)
            #    final_rec_ids = [rec['id'] for rec in final_rec_ids]
            #    radio_playlist, kl_divergence, total_relevance = publishPlaylistRadio(user_id, radio_name, final_rec_ids, str(l))

            radio_playlist = publishPlaylistRadio(user_id, radio_name, [rec['id'] for rec in final_recs])
            
            #Register new playlist radio, model tracks, initialize taboo list and register new recommendations
            registerNewRadio(user_id, playlist, radio_playlist['id'], RECOMMENDATION_PROTOCOL, model_data, final_recs, attr_weights.tolist(), facet_pref_dist, kl_divergence, total_relevance)

            print("Playlist ", radio_name, " generated successfully!")

        else:
            
            # Extract all the current cycle data
            rec_protocol, last_cycle_ts, n_facets, facet_pref_dist, attribute_weights = getCycleData(user_id, playlist['id'])

            print("Refreshing ", radio_name, " playlist with ", rec_protocol, " recommendation protocol...")

            try:
                radio_playlist = sp.playlist(radio_id[0])
                radio_id = radio_playlist['id']
                if radio_playlist['name'] != radio_name:
                    sp.playlist_change_details(radio_id, name=radio_name)
            except:
                radio_check = False
            else:
                radio_check = True
            
            # Extract all the current model data
            model_data = getModelTrackData(user_id, playlist['id'])
            
            # Extract new explicit feedback
            added_tracks, removed_tracks, previous_recs, liked_recs, rejected_recs = extractNewFeedback(user_id, playlist, radio_playlist, 
                                                                                                        radio_check, model_data, last_cycle_ts)
                        
            # Extract recent Spotify streaming history
            recent_stream_count, last_stream_ts = getRecentStreamingHistory(last_cycle_ts)
                         
            # Check if there is enough user feedback to refresh a radio playlist
            added_recs = [track for track in previous_recs if track['feedback'] == "ADDED"]
            n_added = len(added_recs)
            n_liked = len(liked_recs)
            n_rejected = len(rejected_recs)
            if n_added + n_liked + n_rejected == 0:
                raise TypeError("Not enough user feedback was produced to refresh the playlist radio. Canceling RADIO refresh...")
            
            # Update model track data based on new feedback
            model_data, value_cutoffs, current_ts, days_elapsed = getUpdatedModelData(model_data, previous_recs, removed_tracks, recent_stream_count, 
                                                                                      last_stream_ts, last_cycle_ts)

            # Process and prepare new model track data
            newly_added_data, newly_liked_data = getNewModelData(playlist, added_tracks, liked_recs, value_cutoffs, recent_stream_count, 
                                                                 last_stream_ts, last_cycle_ts, current_ts, n_facets)

            # Obtain recommendation pool
            updated_model_relevances = {track['id']: track['relevance'] for track in model_data + newly_added_data + newly_liked_data}
            final_recs = getRecommendations(updated_model_relevances, rec_protocol, playlist['id'])

            # FOR TESTING ONLY:
            # rec_playlist = sp.playlist("7xn1KH7xRDvwoje7zIP4ff")
            # _, final_recs = genTracklistData(rec_playlist)
            # for rec in final_recs:
            #     rec['seed_id'] = "NO SEED"

            # Extract recent facet trend distribution from new feedback and classify new model tracks
            removed_data = getAudioFeatures(removed_tracks)
            rejected_data = getAudioFeatures(rejected_recs)
            model_data, rel_model_data, newly_added_data, newly_liked_data, removed_data, rejected_data = prepareModelData(model_data, newly_added_data, newly_liked_data, 
                                                                                                            removed_data, rejected_data, recent_stream_count, 
                                                                                                            last_stream_ts, len(facet_pref_dist), attribute_weights, 
                                                                                                            current_ts, last_cycle_ts)
            # Extract recent trend distribution from new feedback
            recent_trend_dist = extractRecentTrendDist(rel_model_data, newly_added_data, newly_liked_data, removed_data, rejected_data, len(facet_pref_dist))

            trend_sum = 0.0
            for i, value in enumerate(recent_trend_dist):
                
                if not isinstance(value, (int, float)) or math.isnan(value):
                    recent_trend_dist[i] = 0.0

                trend_sum += recent_trend_dist[i]

            if trend_sum != 0.0:
                pref_dist_update_factor =  days_elapsed / PREF_DIST_UPDATE_FACTOR
                new_facet_pref_dist = [(1 - pref_dist_update_factor) * x + pref_dist_update_factor * y for x, y in zip(facet_pref_dist, recent_trend_dist)]
            
            else:
                new_facet_pref_dist = facet_pref_dist

            print("Old facet preference distribution: ", facet_pref_dist)
            print("New facet preference distribution: ", new_facet_pref_dist)
            
            final_recs, kl_divergence, total_relevance = postRanking(final_recs, new_facet_pref_dist, RADIO_SIZE, model_data, attribute_weights, rec_protocol)

            prev_cycle_stats = [n_added, n_liked, n_rejected]
            updateModel(user_id, playlist, rec_protocol, model_data, added_tracks, removed_tracks, previous_recs, prev_cycle_stats, current_ts, new_facet_pref_dist, kl_divergence, total_relevance)

            if radio_check:
                sp.user_playlist_replace_tracks(user=user_id, playlist_id=radio_id, tracks=[track['id'] for track in final_recs])
            else:
                radio_playlist = publishPlaylistRadio(user_id, radio_name, [track['id'] for track in final_recs])
                updatePlaylistRadioId(user_id, playlist_id, radio_playlist['id'])

            registerRecommendations(user_id, playlist_id, final_recs)

        return jsonify(radio_playlist)
    
    except Exception as e:
        print(str(e))
        return jsonify(playlist)

#####################################################
########### SPOTIFY WEB API COMMUNICATION ###########
#####################################################

# Collect tracks audio feature data through Spotify's API
def getAudioFeatures(track_ids):

    result = []

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)

    for i in range(0, len(track_ids), 100):

        tracks = track_ids[i:i + 100]
        more_features = sp.audio_features(tracks)
        result.extend(more_features)

    return result 

# Filter saved or non-saved tracks
def filterSaved(track_ids, reverse=False):

    if not track_ids:
        return []

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)
    
    in_saved = sp.current_user_saved_tracks_contains(track_ids) 

    filtered_tracks = [track for track, saved in zip(track_ids, in_saved) if not saved]
    
    if reverse:
        return [track for track in track_ids if track not in filtered_tracks]
    else:
        return filtered_tracks  
    
# Collect Spotify tracklist ids and added timestamps
def getTracklistIds(playlist, added_after=None):

    playlist_tracks = playlist['tracks']['items']
    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)

    # Collect all the playlist tracks
    while len(playlist_tracks) < playlist['tracks']['total']:

        extra_tracks = sp.playlist_tracks(playlist["id"], offset=len(playlist_tracks))["items"]
        playlist_tracks.extend(extra_tracks)

    if added_after is None:
        # Returns all track ids without duplicates
        result = list(OrderedDict.fromkeys(track['track']['id'] for track in playlist_tracks if track.get('track') and track['track'].get('id')))

        return result

    elif added_after == 0:
        # Returns all track ids and added dates
        result = {}

        for track in playlist_tracks:
            if track.get('track') and track['track'].get('id'):
                track_id = track['track']['id']

                if track.get('added_at'):
                    result[track_id] = track['added_at']

                else:
                    result[track_id] = None

        return result

    else:
        # Returns all track ids and ids added after the desired timestamp
        old_ids = set()
        new_ids = set()

        for track in playlist_tracks:
            if track.get('track') and track['track'].get('id'):

                track_id = track['track']['id']
                added_ts = time.mktime(datetime.strptime(track.get('added_at', ''), "%Y-%m-%dT%H:%M:%SZ").timetuple())

                if added_ts > added_after:
                    new_ids.add(track_id)

                else:
                    old_ids.add(track_id)

        return list(old_ids), list(new_ids)
    
# Extract user's Spotify streaming history since last cycle
def getRecentStreamingHistory(last_radio_ts):

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)

    streamed_track_ids = []
    streamed_ts = {}

    before_ts = datetime.now().timestamp()
    n_requests = 0

    print("Extracting user's recent streaming history...")
    days, hours, minutes, seconds = getTimeFromTimestamp(last_radio_ts, before_ts)
    print(f"Time Elapsed: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")

    before_ts = int(before_ts * 1000)
    last_radio_ts = int(last_radio_ts * 1000)
    formats = ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z']

    while True:
        recently_streamed = sp.current_user_recently_played(limit=50, before = before_ts)['items']

        n_requests += 1

        for track in recently_streamed:
            for format_string in formats:
                try:
                    ts = int(datetime.strptime(track['played_at'], format_string).timestamp() * 1000)
                    break 
                except ValueError:
                    pass

            if ts < last_radio_ts:
                return Counter(streamed_track_ids), streamed_ts

            if track_id := track['track'].get('id'):
                streamed_track_ids.append(track_id)

                if track_id not in streamed_ts:
                    streamed_ts[track_id] = ts / 1000

        if n_requests == MAX_REC_REQUESTS or len(recently_streamed) == 0:
            break

        before_ts = ts

    print("Recent streaming history analysis completed!")

    return Counter(streamed_track_ids), streamed_ts

#  Extract new user explicit feedback (added/removed tracks, liked/rejected recommendations)
def extractNewFeedback(user_id, playlist, radio_playlist, check_radio, model_data, last_radio_ts):

    print("Extracting Explicit Feedback...")

    # Extract newly added tracks
    playlist_tracks_curr, tracks_added = getTracklistIds(playlist, last_radio_ts)

    # Extract new removed tracks
    playlist_tracks_prev = {track['id'] for track in model_data if track['is_playlist_track']}
    tracks_removed = [track_id for track_id in playlist_tracks_prev if track_id not in playlist_tracks_curr]
    tracks_removed.extend(track_id for track_id in tracks_added if track_id in playlist_tracks_prev)

    # Extracted rejected recommendations
    prev_recs = getRadioTrackIds(user_id, playlist['id'])
    prev_seed_ids = [track['id'] for track in prev_recs]
    rejected_recs = []
    if check_radio:
        radio_tracks_curr = getTracklistIds(radio_playlist)
        added_to_radio = [track for track in radio_tracks_curr if track not in prev_seed_ids]
        if len(added_to_radio) == len(radio_tracks_curr):
           raise TypeError("RADIO playlist has been completely altered.")
        rejected_recs = [id for id in prev_seed_ids if id not in radio_tracks_curr and id not in playlist_tracks_curr and id not in tracks_added]

    # Extract liked recommendations
    liked_recs = filterSaved(prev_seed_ids, reverse=True)
    liked_recs = [track for track in liked_recs if track not in tracks_added and track not in rejected_recs]

    added_recs = [id for id in prev_seed_ids if id in tracks_added]

    for track in prev_recs:
        if track['id'] in added_recs:
            track['feedback'] = "ADDED"
        
        elif track['id'] in liked_recs:
            track['feedback'] = "LIKED"

        elif track['id'] in rejected_recs:
            track['feedback'] = "REJECTED"
        else:
            track['feedback'] = "NONE"

    print("Explicit feedback extraction completed!")
    print("Newly added tracks:", tracks_added, "(amount:", len(tracks_added), ")")
    print("Newly removed tracks:", tracks_removed, "(amount:", len(tracks_removed), ")")
    print("Recommendations:")
    print("-Added:", added_recs, "(amount:", len(added_recs), ")")
    print("-Liked:", liked_recs, "(amount:", len(liked_recs), ")")
    print("-Rejected:", rejected_recs, "(amount:", len(rejected_recs), ")")

    return [tracks_added, tracks_removed, prev_recs, liked_recs, rejected_recs]

# Collect recommendation pool using selected seeds
def getRecommendations(seed_relevances, rec_protocol, playlist_id = ""):
    
    print("Collecting initial recommendation pool...")

    rec_pool = []
    pool_size = RADIO_SIZE if rec_protocol != "TASTES" else RADIO_SIZE * POOL_SIZE_FACTOR
    used_seeds = set()
    seed_ids = list(seed_relevances.keys())
    seed_relevances = list(seed_relevances.values())
    taboo_list = set(getTabooList(getUserId(), playlist_id)) if playlist_id else set(seed_ids[:])
    n_requests = 0

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)
    
    def getNewRecommendations(seed, n_recs):
        recommendations = sp.recommendations(seed_tracks=[seed], limit=n_recs)['tracks']
        return [rec['id'] for rec in recommendations]
    
    def registerRecommendations(track_ids, seed_id):

        track_ids = [id for id in filterSaved(track_ids) if id not in taboo_list]

        taboo_list.update(track_ids)
        used_seeds.add(seed_id)

        for track_id in track_ids: 
            rec_pool.append({"id": track_id, "seed_id": seed_id})
    

    while len(rec_pool) < pool_size and n_requests < MAX_REC_REQUESTS:

        seed_id = random.choices(seed_ids, seed_relevances, k=1)[0]

        while seed_id in used_seeds:
            seed_id = random.choices(seed_ids, seed_relevances, k=1)[0]

        if rec_protocol != "TASTES":

            if n_requests == MAX_REC_REQUESTS - 1:
                seed_id = seed_ids[seed_relevances.index(max(seed_relevances))]
                tracks_left = RADIO_SIZE - len(rec_pool)
                recommendations = getNewRecommendations(seed_id, tracks_left * 5)
                registerRecommendations(recommendations, seed_id)
                
                return rec_pool[:pool_size]
            
            else:
                recommendations = getNewRecommendations(seed_id, 5)
                recommendations = [rec for rec in filterSaved(recommendations) if rec not in taboo_list]
                if len(recommendations) > 0:
                    registerRecommendations([recommendations[0]], seed_id)

        else:
            recommendations = getNewRecommendations(seed_id, REC_REQ_SIZE)
            registerRecommendations(recommendations, seed_id)

        n_requests += 1

    final_rec_pool = getAudioFeatures([rec['id'] for rec in rec_pool])
    for i, rec in enumerate(final_rec_pool):
        rec['seed_id'] = rec_pool[i]['seed_id']

    print("Final recommendation pool collected successfully!")

    return final_rec_pool[:pool_size]

# Publish new RADIO playlist in user's Spotify library
def publishPlaylistRadio(user_id, radio_name, radio_tracks, disc = ""):

    print("Publishing '{}' on '{}''s Spotify Library...".format(radio_name, user_id))

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)
    
    # Create the radio playlist
    radio_playlist = sp.user_playlist_create(user=user_id, name=radio_name, public=False, description=disc)

    # Upload the cover image
    logo_base64 = getLogo()
    sp.playlist_upload_cover_image(radio_playlist['id'], logo_base64)

    # Add tracks to the playlist
    sp.user_playlist_add_tracks(user=user_id, playlist_id=radio_playlist['id'], tracks=radio_tracks)

    print("Playlist '{}' published successfully!".format(radio_playlist['name']))

    return radio_playlist

# Get user's most recently save tracks
def getLastSavedTracks():

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)

    saved_tracks = sp.current_user_saved_tracks(limit=50)['items']
    saved_tracks.extend(sp.current_user_saved_tracks(limit=50, offset=50)['items'])

    return saved_tracks
    
###################################################
########### FACET MODEL DATA PROCESSING ###########
###################################################

# Extract playlist track ids and generate initial relevances
def genTracklistData(playlist):
    
    track_added_dates = getTracklistIds(playlist, 0)
    track_relevance_data = genInitTrackRelevances(track_added_dates)
    track_audio_data = getAudioFeatures(list(track_added_dates.keys()))

    return [track_relevance_data, track_audio_data]
    
# Generate initial model relevances from playlist's tracks' added dates
def genInitTrackRelevances(tracks_added_dates):

    oldest_ts = float('inf')
    newest_ts = float('-inf')

    # Find the oldest and newest timestamps
    for date in tracks_added_dates.values():
        try:
            timestamp = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").timestamp()
        except ValueError:
            continue  # Skip invalid dates

        oldest_ts = min(oldest_ts, timestamp)
        newest_ts = max(newest_ts, timestamp)

    # Calculate total time range
    total_time = newest_ts - oldest_ts  # Avoid division by zero

    if total_time == 0:
        return {track_id: 0.5 for track_id in tracks_added_dates}

    # Calculate relevances
    for track_id, date in tracks_added_dates.items():
        try:
            timestamp = datetime.strptime(date, "%Y-%m-%dT%H:%M:%SZ").timestamp()
        except ValueError:
            timestamp = oldest_ts  # Set invalid dates to oldest timestamp

        normalized_time = (timestamp - oldest_ts) / total_time
        relevancy = 1 / (1 + math.exp(-RELEVANCE_FUNCTION_SLOPE * (normalized_time - RELEVANCE_FUNCTION_MIDPOINT)))
        tracks_added_dates[track_id] = relevancy

    return tracks_added_dates

# Obtain updated relevance values from implicit and explixit feedback
def getUpdatedModelData(model_data, recs, removed_tracks, stream_counts, last_stream_ts, last_cycle_ts):

    updated_relevances = []
    current_ts = datetime.now().timestamp()
    day_ts = 24 * 60 * 60
    total_days_elapsed = (current_ts - last_cycle_ts) / day_ts

    max_relevance = max([track['relevance'] for track in model_data])
    min_relevance = min([track['relevance'] for track in model_data])
    
    for track in model_data[:]:
        if min_relevance == max_relevance:
            track['norm_relevance'] = track['relevance']
        else:
            track['norm_relevance'] = (track['relevance'] - min_relevance) / (max_relevance - min_relevance)

    # First adjust for relevance time decay
    for track in model_data[:]:
        if track['id'] in removed_tracks:
            model_data.remove(track)
            continue

        relevance = track['relevance']
        decay_factor = PLAYLIST_DECAY_FACTOR if track['is_playlist_track'] else MODEL_DECAY_FACTOR
        elapsed_days = total_days_elapsed

        if track['id'] in stream_counts:
            # Way to recover irrelevant tracks
            if relevance < 0.1:
                relevance = 0.1

            relevance *= (1 + STREAM_COUNT_FACTOR * (1 - track['norm_relevance']) * (stream_counts[track['id']] / elapsed_days))

            elapsed_days = (current_ts - last_stream_ts.get(track['id'], last_cycle_ts)) / day_ts

        relevance *= decay_factor ** elapsed_days
        updated_relevances.append(relevance)

    max_value = max(updated_relevances)
    added_rel_cutoff = findValueForPercent(updated_relevances, ADDED_RELEVANCE_CUTOFF)
    liked_rel_cutoff = findValueForPercent(updated_relevances, LIKED_RELEVANCE_CUTOFF)
        
    cutoffs = [liked_rel_cutoff, added_rel_cutoff, max_value]
    
    # If in TASTES we further adjust seed relevances
    if recs:

        success_data = extractSeedSuccess(recs, stream_counts)

        seed_success = {}
        successful_seeds = {}
        unsuccessful_seeds = {}

        # Determine seed success
        for id, success in success_data.items():
            seed_success[id] = (success['added'] - success['rejected'] + 0.5 * success['liked'] + 0.1 * success['streamed']) / success['total']

        # Extract successfull and unseccessful seeds, updating the remaining relevances
        for i, track in enumerate(model_data):

            if track['id'] in seed_success:
                success = seed_success[track['id']]

                if success < 0 and track['seed_success'] < 0 or success <= -2:
                    unsuccessful_seeds[track['id']] = success

                elif success > 0 and track['seed_success'] > 0 or success >= 2:
                    successful_seeds[track['id']] = success

                else:
                    updated_relevances[i] += track['relevance'] * SEED_SUCCESS_FACTOR * (1 - track['norm_relevance']) * success

                track['seed_success'] = success

        successful_seeds = normalizeDictValues(successful_seeds)
        unsuccessful_seeds = normalizeDictValues(unsuccessful_seeds)
        
        max_value = max(updated_relevances)
        successful_rel_cutoff = findValueForPercent(updated_relevances, SUCCESSFUL_RELEVANCE_CUTOFF)
        unsuccessful_rel_cutoff = findValueForPercent(updated_relevances, UNSUCCESSFUL_RELEVANCE_CUTOFF)
        added_rel_cutoff = findValueForPercent(updated_relevances, ADDED_RELEVANCE_CUTOFF)
        liked_rel_cutoff = findValueForPercent(updated_relevances, LIKED_RELEVANCE_CUTOFF)

        # Treat special cases so that the order of relevances is mantained (added > succcessful_seeds > liked > rest > unsuccesful_seeds)
        for i, track in enumerate(model_data):
            if track['id'] in unsuccessful_seeds:
                updated_relevances[i] = unsuccessful_rel_cutoff * unsuccessful_seeds[track['id']]
            elif track['id'] in successful_seeds:
                updated_relevances[i] = successful_rel_cutoff + (added_rel_cutoff - successful_rel_cutoff) * successful_seeds[track['id']]

        cutoffs = [unsuccessful_rel_cutoff, liked_rel_cutoff, successful_rel_cutoff, added_rel_cutoff, max_value]

    # Update model_data with updated relevances
    for i, model_track in enumerate(model_data):
        model_track['relevance'] = updated_relevances[i]

    return [model_data, cutoffs, current_ts, total_days_elapsed]

# Extract seed success from user's feedback on generated recommendations
def extractSeedSuccess(recommendations, stream_data):
    
    # Initialize seed scores
    unique_seeds = set([track['seed_id'] for track in recommendations])
    seed_scores = {seed: {"total": 0, "added": 0, "liked": 0, "rejected": 0, "streamed": 0} for seed in unique_seeds}

    # Adjust seed success according to user feedback to each generated recommendation
    for rec in recommendations:
        seed = rec["seed_id"]
        feedback = rec["feedback"]
        seed_scores[seed]["total"] += 1

        if feedback == "ADDED":
            seed_scores[seed]["added"] += 1

        elif feedback == "REJECTED":
            seed_scores[seed]["rejected"] += 1

        elif feedback == "LIKED":
            seed_scores[seed]["liked"] += 1

        if rec['id'] in stream_data and stream_data[rec['id']] > 1:
            seed_scores[seed]["streamed"] += stream_data[rec['id']] - 1

    return seed_scores

# Get new model track's audio and relevance data
def getNewModelData(playlist, added_tracks, liked_recs, cutoffs, stream_count, last_stream_ts, last_cycle_ts, current_ts, n_facets):

    # Get audio features for added tracks and liked recommendations
    new_track_features = getAudioFeatures(added_tracks + liked_recs)
    split_index = len(added_tracks)

    playlist_added_dates = getTracklistIds(playlist, 0)
    for track_data in new_track_features[:split_index]:
        track_data['added_at'] = time.mktime(datetime.strptime(playlist_added_dates[track_data['id']], "%Y-%m-%dT%H:%M:%SZ").timetuple())

    track_relevances = [(last_stream_ts.get(track['id'], track['added_at']) - last_cycle_ts) / (current_ts - last_cycle_ts) for track in new_track_features[:split_index]]
    track_relevances.extend([(last_stream_ts.get(track['id'], last_cycle_ts) - last_cycle_ts) / (current_ts - last_cycle_ts) for track in new_track_features[split_index:]])
    
    track_relevances = [stream_count.get(track_data['id'], 0) * track_relevances[i] for i, track_data in enumerate(new_track_features)]

    min_relevance_added = min(track_relevances[:split_index]) if track_relevances[:split_index] else 0
    max_relevance_added = max(track_relevances[:split_index]) if track_relevances[:split_index] else 0
    min_relevance_liked = min(track_relevances[split_index:]) if track_relevances[split_index:] else 0
    max_relevance_liked = max(track_relevances[split_index:]) if track_relevances[split_index:] else 0

    added_rel_diff = max_relevance_added - min_relevance_added if max_relevance_added - min_relevance_added > 0 else 1


    for i, relevance in enumerate(track_relevances):
        if i < split_index and (max_relevance_added - min_relevance_added) > 0:
            track_relevances[i] = (relevance - min_relevance_added) / (max_relevance_added - min_relevance_added)
        elif (max_relevance_liked - min_relevance_liked) > 0:
            track_relevances[i] = (relevance - min_relevance_liked) / (max_relevance_liked - min_relevance_liked)
        else:
            track_relevances[i] = 0.5

    # Adjust cutoffs based on n_facets
    if n_facets == 0:
        cutoffs = [0, cutoffs[0], cutoffs[2], cutoffs[2], cutoffs[3]]

    # Process new_track_features for added tracks and liked recommendations
    for i, track_data in enumerate(new_track_features):
        if i < split_index:
            track_data['relevance'] = cutoffs[3] + (cutoffs[4] - cutoffs[3]) * track_relevances[i]
        else:
            track_data['relevance'] = cutoffs[1] + (cutoffs[2] - cutoffs[1]) * track_relevances[i]

    return [new_track_features[:split_index], new_track_features[split_index:]]

#####################################
########### DATA HANDLING ###########
#####################################

# Find value that separates top percentage of data
def findValueForPercent(lst, percentage):

    sorted_lst = sorted(lst)
    percentile_index = int(len(sorted_lst) * percentage)
    value = sorted_lst[percentile_index]

    return value

# Normalize dictionary values
def normalizeDictValues(dictionary):

    if not dictionary:
        return {}

    min_val = min(dictionary.values())
    max_val = max(dictionary.values())

    if not (max_val - min_val) > 0:
        return  {key: 0.5 for key, value in dictionary.items()}

    normalized_dict = {key: (value - min_val) / (max_val - min_val) for key, value in dictionary.items()}
    sorted_normalized_dict = dict(sorted(normalized_dict.items(), key=lambda item: item[1]))

    return sorted_normalized_dict

# Get time elapsed between two timestamps
def getTimeFromTimestamp(start_timestamp, end_timestamp):

    start_time = datetime.fromtimestamp(start_timestamp)
    end_time = datetime.fromtimestamp(end_timestamp)

    # Calculate time difference
    time_difference = end_time - start_time

    # Extract days, hours, minutes, and seconds from the time difference
    days = time_difference.days
    hours, remainder = divmod(time_difference.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return days, hours, minutes, seconds

######################################
########### ASSET HANDLING ###########
######################################

# Get TASTES cover image for RADIO playlists
def getLogo():

    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.jpg')

    with open(logo_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')