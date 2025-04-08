from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    # ... other user attributes

class Session(db.Model):
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)

    user = db.relationship('User', backref=db.backref('sessions', lazy=True))

class LexiconKey(db.Model):
    __tablename__ = 'lexicon_keys'
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(100), nullable=False)
    sentiment_score = db.Column(db.Float)
    category = db.Column(db.String(50))

class Rules(db.Model):
    __tablename__ = 'rules'
    id = db.Column(db.Integer, primary_key=True)
    rule_type = db.Column(db.String(50))
    condition = db.Column(db.Text)
    action = db.Column(db.Text)

class SentimentAnalysis(db.Model):
    __tablename__ = 'sentiment_analyses'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'))
    user_input = db.Column(db.Text)
    result = db.Column(db.String(50))  
    method_used = db.Column(db.String(50))  
    confidence_score = db.Column(db.Float) 

    session = db.relationship('Session', backref=db.backref('sentiment_analyses', lazy=True))

class MLTrainingData(db.Model):
    __tablename__ = 'ml_training_data'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'))
    model_version = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship('Session', backref=db.backref('ml_training_data', lazy=True))

class SessionHistory(db.Model):
    __tablename__ = 'session_history'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'))
    interaction_details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship('Session', backref=db.backref('session_history', lazy=True))

class GoEmotionsData(db.Model):
    __tablename__ = 'go_emotions'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    emotion = db.Column(db.String(50), nullable=False)
    clean_text = db.Column(db.Text)

