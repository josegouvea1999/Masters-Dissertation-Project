import psycopg2
import os

from datetime import datetime
from dotenv import load_dotenv
from .queries import *

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
CONNECTION = psycopg2.connect(DB_URL)

####################################
######## SCHEMAS AND TABLES ########
####################################

# Check if user schema exists 
def checkUserSchema(user_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:
            
            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)
            
            cursor.execute(CHECK_IF_SCHEMA_EXISTS, (sanitized_user_id[1:-1],))

            exists = cursor.fetchone()[0]
    
    return exists

# Initialize user's tables
def initializeTables(user_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)
            cursor.execute(CREATE_USER_SCHEMA.format(sanitized_user_id))
            cursor.execute(CREATE_PLAYLIST_TABLE.format(schema=sanitized_user_id))
            cursor.execute(CREATE_FACET_MODEL_TABLE.format(schema=sanitized_user_id))
            cursor.execute(CREATE_RADIO_TABLE.format(schema=sanitized_user_id))
            cursor.execute(CREATE_TABOO_TABLE.format(schema=sanitized_user_id))
            cursor.execute(CREATE_CYCLE_TABLE.format(schema=sanitized_user_id))

    CONNECTION.commit()

####################################
########## PLAYLIST TABLE ##########
####################################
    
# Get all of the IDs of user's playlists which have a RADIO
def getUsersRadioedPlaylists(user_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:
            
            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_ALL_PLAYLIST_IDS_BY_USER.format(schema=sanitized_user_id))
            playlist_ids = cursor.fetchall()
            playlist_ids = [id[0] for id in playlist_ids]

    return playlist_ids
 
# Get all the user's RADIO playlist IDs   
def getAllRadioPlaylists(user_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:
            
            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_ALL_RADIO_IDS_BY_USER.format(schema=sanitized_user_id))
            radio_ids = cursor.fetchall()
            radio_ids = [id[0] for id in radio_ids]

    return radio_ids

# Get a playlist's RADIO playlist ID
def getPlaylistRadio(user_id, playlist_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)
            
            cursor.execute(GET_RADIO_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist_id,))
            radio_id = cursor.fetchone()
    
    return radio_id

# Get current cycle data (last cycle timestamp, facet preference distribution, number of facets and attribute weights)
def getCycleData(user_id, playlist_id):
    
    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_CYCLE_DATA_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist_id,))
            cycle_data = cursor.fetchone()

    rec_protocol, last_radio_ts, n_facets = cycle_data[:3]
    pref_dist = cycle_data[3]
    attribute_weights = list(cycle_data[4:])

    return rec_protocol, last_radio_ts, n_facets, pref_dist, attribute_weights

# Update playlist's radio ID to a new one
def updatePlaylistRadioId(user_id, playlist_id, new_radio_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)
            
            cursor.execute(UPDATE_RADIO_ID.format(schema=sanitized_user_id), (new_radio_id, playlist_id))
    
    CONNECTION.commit()

#######################################
########## FACET MODEL TABLE ##########
#######################################
    
# Get all current model track data
def getModelTrackData(user_id, playlist_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_MODEL_TRACK_DATA.format(schema=sanitized_user_id), (playlist_id,))
            model_data = cursor.fetchall()

    track_data = []

    for row in model_data:
        track_data.append({'id': row[0],
                           'facet': row[1],
                           'relevance': row[2],
                           'sim_threshold': row[3],
                           'seed_success': row[4],
                           'is_playlist_track': row[5],
                           'danceability': row[6],
                           'acousticness': row[7],
                           'energy': row[8],
                           'instrumentalness': row[9],
                           'liveness': row[10],
                           'loudness': row[11],
                           'speechiness': row[12],
                           'tempo': row[13],
                           'valence': row[14],
                           'time_signature': row[15],
                           'mode': row[16],
                           'key': row[17]})

    return track_data

# Register new facet model (register new playlist RADIO and facet model tracks)
def registerNewRadio(user_id, playlist, radio_id, rec_protocol, track_features, recommendations, attr_weights, facet_pref_dist, kl_div, total_rel):

    n_facets = len(facet_pref_dist) if rec_protocol == "TASTES" else 0

    with CONNECTION:
        with CONNECTION.cursor() as cursor:
            
            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            current_ts = datetime.now().timestamp()

            cursor.execute(ADD_NEW_PLAYLIST.format(schema=sanitized_user_id), (
                                playlist['id'],
                                playlist['name'],
                                rec_protocol,
                                radio_id, 
                                n_facets,
                                facet_pref_dist,
                                current_ts,
                                current_ts,
                                0,
                                attr_weights[0],
                                attr_weights[1],
                                attr_weights[2],
                                attr_weights[3],
                                attr_weights[4],
                                attr_weights[5],
                                attr_weights[6],
                                attr_weights[7],
                                attr_weights[8],
                                attr_weights[9],
                                attr_weights[10]))

            for track in track_features:

                cursor.execute(INSERT_FACET_TRACK.format(schema=sanitized_user_id), (
                                    track["id"], 
                                    playlist["id"],
                                    track["facet"],
                                    track["relevance"],
                                    0,
                                    track["sim_threshold"],
                                    0,
                                    True,
                                    False,
                                    track["danceability"],
                                    track["acousticness"],
                                    track["energy"],
                                    track["instrumentalness"],
                                    track["liveness"],
                                    track["loudness"], 
                                    track["speechiness"],
                                    track["tempo"],
                                    track["valence"],
                                    track["time_signature"],
                                    track["mode"],
                                    track["key"]))

                cursor.execute(INSERT_TRACK_TABOO.format(schema=sanitized_user_id), (
                                    track["id"],
                                    playlist["id"],
                                    0))
    
    registerCycle(user_id, playlist["id"], rec_protocol, 0, facet_pref_dist, kl_div, total_rel)
    registerRecommendations(user_id, playlist["id"], recommendations)

# Update facet model (remove removed tracks, add liked and addded tracks, update model relevance data, update cycle data)
def updateModel(user_id, playlist, rec_protocol, model_data, added_track_ids, removed_track_ids, previous_recs, prev_cycle_stats, current_ts, facet_pref_dist, kl_div, total_rel):

    print("Updating Model Database...")

    prev_rec_ids = [rec['id'] for rec in previous_recs]

    # Normalize model relevances
    model_relevances = [track['relevance'] for track in model_data]
    max_rel = max(model_relevances)
    max_rel = max(max_rel, 1)

    # Update track relevances in the model data
    for i, track in enumerate(model_data):
        track['relevance'] = track['relevance'] / max_rel

    # Database operations
    with CONNECTION:
        with CONNECTION.cursor() as cursor:
            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_COUNTER_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist['id'],))
            n_cycle = cursor.fetchone()[0] + 1

            # Remove removed tracks from the model
            for id in removed_track_ids:
                cursor.execute(REMOVE_MODEL_TRACK_BY_ID.format(schema=sanitized_user_id), (id, playlist['id']))

            # Update model track relevances, similarity thresholds, and seed success
            for track in model_data:

                if track['id'] in added_track_ids or track['id'] in prev_rec_ids:

                    is_recommendation = track['id'] in prev_rec_ids
                    is_playlist_track = track['id'] in added_track_ids

                    cursor.execute(INSERT_FACET_TRACK.format(schema=sanitized_user_id), (track['id'], 
                                                                                        playlist['id'], 
                                                                                        track['facet'],
                                                                                        track['relevance'],
                                                                                        0,
                                                                                        track['sim_threshold'],
                                                                                        n_cycle,
                                                                                        is_playlist_track,
                                                                                        is_recommendation,
                                                                                        *map(track.get, ["danceability", "acousticness", "energy", "instrumentalness", 
                                                                                                         "liveness", "loudness", "speechiness", "tempo", "valence", 
                                                                                                         "time_signature", "mode", "key"])))
                    
                    if not is_recommendation:
                        cursor.execute(INSERT_TRACK_TABOO.format(schema=sanitized_user_id), (track['id'], playlist['id'], n_cycle))

                else:

                    cursor.execute(UPDATE_MODEL_TRACK.format(schema=sanitized_user_id), (track['relevance'], 
                                                                                        track['sim_threshold'], 
                                                                                        track['seed_success'], 
                                                                                        track['id'],
                                                                                        playlist['id']))

            # Register new user feedback to the last recommendations
            for track in previous_recs:
                cursor.execute(UPDATE_RECS_FEEDBACK.format(schema=sanitized_user_id),(track["feedback"],
                                                                                      track['id'],
                                                                                      playlist['id'],
                                                                                      n_cycle - 1))

            # Update playlist table
            cursor.execute(UPDATE_PLAYLIST_CYCLE_DATA.format(schema=sanitized_user_id),(playlist['name'],
                                                                                        facet_pref_dist,
                                                                                        current_ts,
                                                                                        playlist['id']))
            
    registerCycle(user_id, playlist["id"], rec_protocol, n_cycle, facet_pref_dist, kl_div, total_rel, prev_cycle_stats)

    CONNECTION.commit()

    print("Model database updated successfully!")

    return n_cycle

#################################
########## RADIO TABLE ##########
#################################
    
# Refresh recommendations and add to taboo
def registerRecommendations(user_id, playlist_id, recommendations):
    
    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_COUNTER_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist_id,))
            n_cycle = cursor.fetchone()[0]
                                       
            for rec in recommendations:
                
                cursor.execute(INSERT_RECOMMENDATION.format(schema=sanitized_user_id), (rec["id"], 
                                                                                        playlist_id, 
                                                                                        rec["seed_id"], 
                                                                                        n_cycle))

                cursor.execute(INSERT_TRACK_TABOO.format(schema=sanitized_user_id), (rec["id"], playlist_id, n_cycle))

    CONNECTION.commit()
    CONNECTION.close()

# Get last recommendation IDs and seeds
def getRadioTrackIds(user_id, playlist_id):

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_COUNTER_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist_id,))
            n_cycle = cursor.fetchone()[0]

            cursor.execute(GET_RECS_BY_PLAYLIST_ID.format(schema=sanitized_user_id), (playlist_id, n_cycle))
            prev_rec_data = cursor.fetchall()

    if not prev_rec_data:
        return []
    
    else:
        prev_rec_ids, prev_rec_seeds = zip(*prev_rec_data)

        return [{"id": prev_rec_ids[i], "seed_id": prev_rec_seeds[i]} for i in range(len(prev_rec_ids))]

#################################
########## TABOO TABLE ##########
#################################

# Get all tracks in taboo list
def getTabooList(user_id, playlist_id):
    
    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(GET_ALL_TABOO_TRACKS.format(schema=sanitized_user_id), (playlist_id,))
            taboo_ids = cursor.fetchall()
            taboo_ids = [id[0] for id in taboo_ids]

    return taboo_ids

#################################
########## CYCLE TABLE ##########
#################################

def registerCycle(user_id, playlist_id, rec_protocol, n_cycle, pref_dist, kl_divergence, total_relevance, prev_cycle_stats = []):

    rec_protocol = "SPOTIFY" if rec_protocol != "TASTES" else rec_protocol

    with CONNECTION:
        with CONNECTION.cursor() as cursor:

            sanitized_user_id = psycopg2.extensions.quote_ident(user_id, cursor)

            cursor.execute(INSERT_CYCLE_DATA.format(schema=sanitized_user_id), (playlist_id, rec_protocol, n_cycle, 
                                                                                pref_dist, kl_divergence, total_relevance))
            
            if prev_cycle_stats != []:
                cursor.execute(REGISTER_CYCLE_FEEDBACK.format(schema=sanitized_user_id), (prev_cycle_stats[0], 
                                                                                          prev_cycle_stats[1], 
                                                                                          prev_cycle_stats[2],
                                                                                          playlist_id, 
                                                                                          rec_protocol, 
                                                                                          n_cycle - 1))