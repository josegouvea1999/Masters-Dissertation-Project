import requests
import urllib.parse

from flask import Flask, jsonify, redirect, request
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = "my secret key"

db = SQLAlchemy(app)

class Token(db.Model):
    id = db.Column(db.String, primary_key=True)
    access_token = db.Column(db.String(200), nullable=False)
    refresh_token = db.Column(db.String(200), nullable=False)
    expires_at = db.Column(db.Float, nullable=False)

CORS(app, supports_credentials=True)

CLIENT_ID = '95191795bc7340a99072a2c91579fb18'
CLIENT_SECRET = '549e045b4aef4897afbe964beb146886'
REDIRECT_URI = 'http://192.168.1.114:5000/callback'

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1/'

TOKEN_ID = 1

@app.route('/login')
@cross_origin()
def login():
    scope = 'user-read-private user-read-email playlist-read-private playlist-modify'

    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'scope': scope,
        'redirect_uri': REDIRECT_URI,
        'show_dialog': True # user has to re-login every time
    }

    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    return jsonify(auth_url)

@app.route('/callback')
@cross_origin()
def callback():
    if 'error' in request.args:
        return jsonify({'error': request.args['error']})
    
    if 'code' in request.args:
        req_body = {
            'code': request.args['code'],
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }

        response = requests.post(TOKEN_URL, data=req_body)
        token_info = response.json()

        token = Token(id = TOKEN_ID, access_token = token_info['access_token'], 
                      refresh_token = token_info['refresh_token'], 
                      expires_at = datetime.now().timestamp() + token_info['expires_in'])
       
        Token.query.delete()
        db.session.add(token)
        db.session.commit()

        return redirect('exp://192.168.1.114:8081/--/redirect')
    

@app.route('/refresh-token')
@cross_origin()
def refresh_token():

    token_to_update = Token.query.filter_by(id = TOKEN_ID).first()

    if token_to_update is None:
        print('oh boy')
        redirect(login()) # support this on the frontend

    if datetime.now().timestamp() > token_to_update.expires_at:
        req_body = {
            'grant_type':'refresh_token',
            'refresh_token': token_to_update.refresh_token,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
        
        response = requests.post(TOKEN_URL, data=req_body)
        new_token_info = response.json()


        token_to_update.access_token = new_token_info['access_token']
        token_to_update.expires_at = datetime.now().timestamp() + new_token_info['expires_in']
        
        return 'success'

@app.route('/check_token')
def check_refresh(token):

    if token is None:
        redirect(login()) # support this on the frontend

    if datetime.now().timestamp() > token.expires_at:
        refresh_token()

    return 'Token up to date'

@app.route('/playlists')
@cross_origin()
def get_playlists():

    token_to_update = Token.query.filter_by(id = TOKEN_ID).first()

    try:
        check_refresh(token_to_update)
    except:
        raise TypeError("Issue refreshing access token")
    else:
        access_token = token_to_update.access_token
    
        headers = {
            'Authorization': f"Bearer {access_token}"
        }

        response = requests.get(API_BASE_URL + 'me/playlists', headers=headers)
        playlists = response.json()

        return jsonify(playlists)


@app.route('/profile')
@cross_origin()
def get_profile():

    token_to_update = Token.query.filter_by(id = TOKEN_ID).first()

    try:
        check_refresh(token_to_update)
    except:
        raise TypeError("Issue refreshing access token")
    else:

        access_token = token_to_update.access_token

        headers = {
            'Authorization': f"Bearer {access_token}"
        }

        response = requests.get(API_BASE_URL + 'me', headers=headers)

        return jsonify(response.json())

@app.route('/refresh/<id>')
@cross_origin()
def refresh(id):

    token_to_update = Token.query.filter_by(id = TOKEN_ID).first()

    try:
        check_refresh(token_to_update)
    except:
        raise TypeError("Issue refreshing access token")
    else:

        access_token = token_to_update.access_token

        headers = {
            'Authorization': f"Bearer {access_token}"
        }

        response = requests.get(API_BASE_URL + 'playlists/' + id + '/images', headers=headers)

        print(response.url)
        return jsonify(response.url)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        app.run(host='0.0.0.0', debug=True)
