from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Token(db.Model):
    id = db.Column(db.String, primary_key=True)
    access_token = db.Column(db.String(200), nullable=False)
    refresh_token = db.Column(db.String(200), nullable=False)
    expires_at = db.Column(db.Float, nullable=False)

class CurrentUser(db.Model):
    id = db.Column(db.String, primary_key=True)