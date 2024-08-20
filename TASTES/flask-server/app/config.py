import os

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SECRET_KEY = "my secret key"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DEBUG = True
