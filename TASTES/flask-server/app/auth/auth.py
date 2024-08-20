import os
import spotipy

from app import app
from flask import jsonify, url_for, request, redirect
from flask_cors import cross_origin
from spotipy.oauth2 import SpotifyOAuth
from datetime import datetime
from .models import db, Token, CurrentUser 
from dotenv import load_dotenv

load_dotenv()
frontend_redirect = 'exp://' + os.getenv("LOCALHOST") + ':8081/--/redirect'

@app.route('/login', methods=["GET"])
@cross_origin()
def login():
        
    auth_url = createSpotifyOauth().get_authorize_url()

    return jsonify(auth_url)

@app.route('/callback', methods=["GET"])
@cross_origin()
def callback():
    
    if 'error' in request.args:
        
        return jsonify({'error': request.args['error']})
    
    elif 'code' in request.args:

        token_info = createSpotifyOauth().get_access_token(request.args['code'])
        user_id = spotipy.Spotify(auth=token_info['access_token']).me()['id']

        current_user_token = Token.query.filter_by(id = user_id).first()
    
        if current_user_token is None:
            
            new_user_token = Token(id = user_id, 
                               access_token = token_info['access_token'], 
                               refresh_token = token_info['refresh_token'], 
                               expires_at = datetime.now().timestamp() + token_info['expires_in'])

            db.session.add(new_user_token)
            
        else:
            
            current_user_token.access_token = token_info['access_token']
            current_user_token.refresh_token = token_info['refresh_token']
            current_user_token.expires_at = datetime.now().timestamp() + token_info['expires_in']

        CurrentUser.query.delete()
        current_user = CurrentUser(id = user_id)
            
        db.session.add(current_user)
            
        db.session.commit()
        
        return redirect(frontend_redirect)
    
    else:
        
        return jsonify({'auth_token_missing': request.args})
    
@app.route('/profile', methods=["GET"])
@cross_origin()
def getProfile():

    token = getCurrentUserToken()
    
    sp = spotipy.Spotify(auth=token.access_token)    

    response = sp.current_user()

    return jsonify(response)

def createSpotifyOauth():
        
    return SpotifyOAuth(
        client_id = os.getenv("CLIENT_ID"),  
        client_secret = os.getenv("CLIENT_SECRET"),  
        redirect_uri = url_for('callback', _external=True),
        scope = os.getenv("SCOPES"),
        show_dialog= False,
        cache_path=os.getenv('SPOTIFY_USERNAME'),
        username=os.getenv('SPOTIFY_USERNAME'),
        )

    
def getCurrentUserToken():
    
    current_user_id = getUserId()
    
    current_user_token = Token.query.filter_by(id = current_user_id).first()
    
    if current_user_token is None:
        raise TypeError("Token not found")
    
    elif current_user_token.expires_at < datetime.now().timestamp(): 
        try:
            spotipy.Spotify(auth=current_user_token.access_token).me()
    
        except:
        
            new_token_info = createSpotifyOauth().refresh_access_token(current_user_token.refresh_token)

            current_user_token.access_token = new_token_info['access_token']
            current_user_token.expires_at = datetime.now().timestamp() + new_token_info['expires_in']
        
            try:
                db.session.commit()
            except:
                raise TypeError("Issue refreshing access token")
            else:
                return current_user_token   
        
    else:
        return current_user_token
    
def getUserId():

    return CurrentUser.query.first().id
    