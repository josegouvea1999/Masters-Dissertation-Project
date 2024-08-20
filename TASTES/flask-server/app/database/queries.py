#######################################
############### SCHEMAS ###############
#######################################

CREATE_USER_SCHEMA =""" CREATE SCHEMA IF NOT EXISTS {}"""

CHECK_IF_SCHEMA_EXISTS = """SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.schemata
                            WHERE schema_name = (%s)
                            );"""

##############################################
############### PLAYLIST TABLE ###############
##############################################

CREATE_PLAYLIST_TABLE = """CREATE TABLE IF NOT EXISTS {schema}.playlists (
                                    playlist_id VARCHAR(255) PRIMARY KEY,
                                    playlist_name VARCHAR(255),
                                    rec_protocol VARCHAR(255),
                                    radio_id VARCHAR(255),
                                    n_facets INTEGER NOT NULL,
                                    pref_dist FLOAT[] NOT NULL,
                                    init_ts NUMERIC NOT NULL,
                                    last_radio_ts NUMERIC NOT NULL, 
                                    cycle_counter INTEGER NOT NULL,
                                    w_danceability NUMERIC NOT NULL,
                                    w_acousticness NUMERIC NOT NULL,
                                    w_energy NUMERIC NOT NULL,
                                    W_instrumentalness NUMERIC NOT NULL,
                                    w_liveness NUMERIC NOT NULL,
                                    w_loudness NUMERIC NOT NULL,
                                    w_speechiness NUMERIC NOT NULL,
                                    w_tempo NUMERIC NOT NULL,
                                    w_valence NUMERIC NOT NULL,
                                    w_time_signature NUMERIC NOT NULL,
                                    w_key NUMERIC NOT NULL);"""

GET_ALL_PLAYLIST_IDS_BY_USER = """SELECT playlist_id 
                                    FROM {schema}.playlists;"""
                                
GET_ALL_RADIO_IDS_BY_USER = """SELECT radio_id 
                                FROM {schema}.playlists;"""
                                
GET_RADIO_BY_PLAYLIST_ID = """SELECT radio_id 
                                FROM {schema}.playlists 
                                WHERE playlist_id = (%s);"""

GET_COUNTER_BY_PLAYLIST_ID = """SELECT cycle_counter 
                                FROM {schema}.playlists 
                                WHERE playlist_id = (%s);"""

GET_CYCLE_DATA_BY_PLAYLIST_ID = """SELECT rec_protocol,
                                        CAST(last_radio_ts AS float),
                                        n_facets,
                                        pref_dist,
                                        CAST(w_danceability AS float),
                                        CAST(w_acousticness AS float),
                                        CAST(w_energy AS float),
                                        CAST(W_instrumentalness AS float),
                                        CAST(w_liveness AS float),
                                        CAST(w_loudness AS float),
                                        CAST(w_speechiness AS float),
                                        CAST(w_tempo AS float),
                                        CAST(w_valence AS float),
                                        CAST(w_time_signature AS float),
                                        CAST(w_key AS float)
                                    FROM {schema}.playlists 
                                    WHERE playlist_id = (%s);"""

ADD_NEW_PLAYLIST = """INSERT INTO {schema}.playlists  
                            (playlist_id, playlist_name, rec_protocol, radio_id, n_facets, pref_dist, init_ts, last_radio_ts, cycle_counter, 
                                w_danceability, w_acousticness, w_energy, W_instrumentalness, w_liveness, w_loudness, 
                                w_speechiness, w_tempo, w_valence, w_time_signature, w_key) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""

UPDATE_PLAYLIST_CYCLE_DATA = """UPDATE {schema}.playlists
                                    SET playlist_name = (%s),
                                        pref_dist = (%s),
                                        last_radio_ts = (%s),
                                        cycle_counter = cycle_counter + 1
                                    WHERE playlist_id = (%s);"""

UPDATE_RADIO_ID = """UPDATE {schema}.playlists 
                        SET radio_id = (%s) 
                        WHERE playlist_id = (%s);"""

#########################################################
################### FACET MODEL TABLE ###################
#########################################################

CREATE_FACET_MODEL_TABLE = """CREATE TABLE IF NOT EXISTS {schema}.facet_model (
                                    track_id VARCHAR(255) NOT NULL,
                                    playlist_id VARCHAR(255) NOT NULL,
                                    facet VARCHAR(1) NOT NULL,
                                    relevance NUMERIC NOT NULL,
                                    sim_threshold NUMERIC NOT NULL,
                                    seed_success NUMERIC NOT NULL,
                                    added_cycle INTEGER NOT NULL,
                                    is_playlist_track BOOLEAN NOT NULL,
                                    is_recommendation BOOLEAN NOT NULL,
                                    danceability NUMERIC NOT NULL,
                                    acousticness NUMERIC NOT NULL,
                                    energy NUMERIC NOT NULL,
                                    instrumentalness NUMERIC NOT NULL,
                                    liveness NUMERIC NOT NULL,
                                    loudness NUMERIC NOT NULL,
                                    speechiness NUMERIC NOT NULL,
                                    tempo NUMERIC NOT NULL,
                                    valence NUMERIC NOT NULL,
                                    time_signature INTEGER NOT NULL,
                                    mode INTEGER NOT NULL,
                                    key INTEGER NOT NULL,
                                    PRIMARY KEY(track_id, playlist_id));"""

GET_MODEL_TRACK_DATA = """SELECT track_id,
                                facet,
                                CAST(relevance AS float),
                                CAST(sim_threshold AS float),
                                CAST(seed_success AS float),
                                is_playlist_track,
                                CAST(danceability AS float),
                                CAST(acousticness AS float),
                                CAST(energy AS float),
                                CAST(instrumentalness AS float),
                                CAST(liveness AS float),
                                CAST(loudness AS float),
                                CAST(speechiness AS float),
                                CAST(tempo AS float),
                                CAST(valence AS float),
                                time_signature,
                                mode,
                                key
                            FROM {schema}.facet_model
                            WHERE playlist_id = (%s);"""

INSERT_FACET_TRACK = """INSERT INTO {schema}.facet_model
                            (track_id, playlist_id, facet, relevance, seed_success, sim_threshold, 
                                added_cycle, is_playlist_track, is_recommendation, danceability, 
                                acousticness, energy, instrumentalness, liveness, loudness, 
                                speechiness, tempo, valence, time_signature, mode, key) 
                        VALUES 
                            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (track_id, playlist_id) DO UPDATE
                        SET 
                            facet = EXCLUDED.facet,
                            relevance = EXCLUDED.relevance,
                            seed_success = facet_model.seed_success,
                            sim_threshold = EXCLUDED.sim_threshold,
                            added_cycle = facet_model.added_cycle,
                            is_playlist_track = EXCLUDED.is_playlist_track,
                            is_recommendation = facet_model.is_recommendation,
                            danceability = EXCLUDED.danceability,
                            acousticness = EXCLUDED.acousticness,
                            energy = EXCLUDED.energy,
                            instrumentalness = EXCLUDED.instrumentalness,
                            liveness = EXCLUDED.liveness,
                            loudness = EXCLUDED.loudness,
                            speechiness = EXCLUDED.speechiness,
                            tempo = EXCLUDED.tempo,
                            valence = EXCLUDED.valence,
                            time_signature = EXCLUDED.time_signature,
                            mode = EXCLUDED.mode,
                            key = EXCLUDED.key;"""

REMOVE_MODEL_TRACK_BY_ID = """DELETE FROM {schema}.facet_model
                                WHERE track_id = (%s) AND playlist_id = (%s);"""

UPDATE_MODEL_TRACK = """UPDATE {schema}.facet_model
                            SET relevance = (%s), sim_threshold = (%s), seed_success = (%s)
                            WHERE track_id = (%s) AND playlist_id = (%s);""" 

###################################
########### RADIO TABLE ###########
###################################

CREATE_RADIO_TABLE = """CREATE TABLE IF NOT EXISTS {schema}.radio (
                                track_id VARCHAR(255) NOT NULL,
                                playlist_id VARCHAR(255) NOT NULL,
                                seed_id VARCHAR(255) NOT NULL,
                                n_cycle INTEGER NOT NULL,
                                user_feedback VARCHAR(255) DEFAULT 'NONE',
                                PRIMARY KEY(track_id, playlist_id, n_cycle));"""

GET_RECS_BY_PLAYLIST_ID = """SELECT track_id, seed_id  
                                FROM {schema}.radio
                                WHERE playlist_id = (%s) AND n_cycle = (%s)"""


INSERT_RECOMMENDATION = """INSERT INTO {schema}.radio 
                                (track_id, playlist_id, seed_id, n_cycle) 
                                VALUES (%s, %s, %s, %s);"""

UPDATE_RECS_FEEDBACK = """UPDATE {schema}.radio
                            SET user_feedback = (%s)
                            WHERE track_id = (%s) AND playlist_id = (%s) AND n_cycle = (%s)"""

###################################
########### TABOO TABLE ###########
###################################

CREATE_TABOO_TABLE = """CREATE TABLE IF NOT EXISTS {schema}.taboo (
                        track_id VARCHAR(255) NOT NULL,
                        playlist_id VARCHAR(255) NOT NULL,
                        cycle_added INTEGER NOT NULL,
                        PRIMARY KEY(track_id, playlist_id));"""

GET_ALL_TABOO_TRACKS = """SELECT track_id  
                            FROM {schema}.taboo 
                            WHERE playlist_id = (%s)"""

INSERT_TRACK_TABOO = """INSERT INTO {schema}.taboo 
                            (track_id, playlist_id, cycle_added) 
                            VALUES (%s, %s, %s)
                        ON CONFLICT (track_id, playlist_id) DO NOTHING;"""

###################################
########### CYCLE TABLE ###########
###################################

CREATE_CYCLE_TABLE = """CREATE TABLE IF NOT EXISTS {schema}.cycle (
                                    playlist_id VARCHAR(255) NOT NULL,
                                    rec_protocol VARCHAR(255) NOT NULL,
                                    cycle_counter INTEGER NOT NULL,
                                    pref_dist FLOAT[] NOT NULL,
                                    kl_divergence NUMERIC NOT NULL,
                                    total_relevance NUMERIC NOT NULL,
                                    added_recs INTEGER,
                                    liked_recs INTEGER,
                                    rejected_recs INTEGER,
                                    PRIMARY KEY(playlist_id, rec_protocol, cycle_counter));"""

INSERT_CYCLE_DATA = """INSERT INTO {schema}.cycle
                            (playlist_id, rec_protocol, cycle_counter, pref_dist, kl_divergence, total_relevance)
                            VALUES (%s, %s, %s, %s, %s, %s);"""

REGISTER_CYCLE_FEEDBACK = """UPDATE {schema}.cycle
                                SET added_recs = (%s),
                                    liked_recs = (%s),
                                    rejected_recs = (%s)
                                WHERE playlist_id = (%s) AND rec_protocol = (%s) AND cycle_counter = (%s);"""
                        