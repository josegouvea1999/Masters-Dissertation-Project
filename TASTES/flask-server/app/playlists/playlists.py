import spotipy

from app import app
from flask import jsonify
from flask_cors import cross_origin
from ..auth.auth import getCurrentUserToken
from ..database.database import getAllRadioPlaylists, checkUserSchema, getUsersRadioedPlaylists

@app.route('/playlists', methods=["GET"])
@cross_origin()
def getPlaylists():

    token = getCurrentUserToken()
    sp = spotipy.Spotify(auth=token.access_token)    
    user_id = sp.me()['id']
    
    playlists = sp.current_user_playlists()
    
    # playlists['items'] = [item for item in playlists['items'] if item['owner']['id'] == user_id]
    
    if checkUserSchema(user_id):
        
        radio_ids = getAllRadioPlaylists(user_id)
        
        playlists['items'] = [item for item in playlists['items'] if item['id'] not in radio_ids]

    return jsonify(playlists)

@app.route('/tastes_playlists', methods=["GET"])
@cross_origin()
def getRadioedPlaylists():
    
    token = getCurrentUserToken()
    
    sp = spotipy.Spotify(auth=token.access_token)
    
    user_id = sp.me()['id']
    
    result = []
    
    if checkUserSchema(user_id):
        
        user_playlists = getUsersRadioedPlaylists(user_id)
        
        for playlist_id in user_playlists:
            
            playlist = sp.playlist(playlist_id)
            result.append(playlist)

    return jsonify(result)