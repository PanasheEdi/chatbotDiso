import pytest
from unittest.mock import patch
from app import app
from werkzeug.security import generate_password_hash
from models import db, User
from flask import jsonify
import jwt

@pytest.fixture
def client():
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Shaun2009@localhost/test_db'
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def create_user(client):
    with app.app_context():
        user = User.query.filter_by(
            email="Unittest@example.com").first() 
        if not user:  
            user = User(
                email="Unittest@example.com",
                password=generate_password_hash("password123")
            )
            db.session.add(user)
            db.session.commit()

"""
#Test 1- registration route (POST/register)
def test_register(client):
    email = 'new_user@example.com' 
    response = client.post('/register', data=dict(
        email='new_Mainuser@example.com',
        password='password123',
        password_confirmation = 'password123'
    ))
    #checking if email already exists 
    if User.query.filter_by(
        email=email).first():
        return jsonify({"error":"Email already registered"}), 409
    
    new_user = User(
        email=email,
        password=generate_password_hash(password))
    db.session.add(new_user)
    db.session.commit()

    assert response.status_code == 201
    assert b'User registered successfully' in response.data

#Test 2 - login route (POST/login)
def test_login(client, create_user):
    response=client.post('login', json=dict(
        email = 'new_Mainuser@example.com',
        password = 'password123'
    ))

    assert response.status_code == 200
    assert b'Login success' in  response.data
    assert b'token' in response.json
"""
#Test 3 -   chat post route 
def test_chat_get(client):
    response = client.get('/chat')
    assert response.status_code == 200
    assert response.json['message'] == 'How Are You Feeling Today? 😊'

#Test 4 - handling empty input in chat POST
def test_chat_post_empty(client):
    response = client.post('/chat', json=dict(
    message=""
    ))
    assert response.status_code==400
    assert b"Missing 'message' in request" in response.data

#Test 5 - testing chat post with mocked sentiment analysis 
@patch ('app.NRCAnalyser.generate_sentiment')
def test_chat_post_with_sentiment(mocked_analyser, client, create_user):
    mocked_analyser.return_value = {
        'dominant_category':'joy',
        'dominant_emotion_score':0.9
    }
    response = client.post('/chat',json=dict(
        message="I feel happy today"
    ))
    assert response.status_code == 200
    assert b"I'm glad you're feeling happy! How can I help you today?" in response.json['response']

    assert response.json['sentiment']['dominant_category'] == 'joy'
    assert response.json['sentiment']['dominant_emotion_score'] == 0.9
