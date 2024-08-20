from flask import Flask
from flask_cors import CORS
from app.auth.models import db
from .config import Config
import atexit
import os

app = Flask(__name__)

app.config.from_object(Config)
CORS(app, supports_credentials=True)

db.init_app(app)

from app.auth import auth
from app.database import database
from app.playlists import playlists
from app.spotify import spotify