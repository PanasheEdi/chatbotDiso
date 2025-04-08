import os
import sys
from nrc_analyser import NRCAnalyser
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from models import db, User, Session, SentimentAnalysis
from flask import send_from_directory
import jwt


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Shaun2009@localhost/Chatbot_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
app.config['SECRET_KEY']=os.environ.get('JWT_SECRET','YourSecretKey')

db.init_app(app)

# Initialize DB & Migrations
migrate = Migrate(app, db)

# Ensure tables are created (only for first-time setup)
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return redirect(url_for('login'))

# Serve HTML templates from src/templates
@app.route('/templates/navigationBar.html')
def get_navigation_template():
    return send_from_directory('src/templates', 'navigationBar.html')

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        data = request.form
        email = data.get('email')
        password = data.get('password')
        password_confirmation = data.get('password_confirmation')

        if not email or not password or not password_confirmation:
            return jsonify({'error': "Email, password, and confirmation are required"}), 400

        if password != password_confirmation:
            return jsonify({'error': "Passwords do not match"}), 400

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return jsonify({'error': 'Email already exists'}), 409

        hashed_password = generate_password_hash(password)

        new_user = User(email=email, password=hashed_password)

        db.session.add(new_user)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f"An error occurred while saving the user: {str(e)}"}), 500
        finally:
            db.session.remove() 

        return jsonify({'message': 'User registered successfully','redirect':'/login'}), 201


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            token = jwt.encode({
                'user_id': user.id,
                'email': user.email,
                'exp': datetime.now(timezone.utc) + timedelta(minutes=30)  # Fix here
            }, app.config['SECRET_KEY'], algorithm='HS256')
            return jsonify({
                'message': 'Login success',
                'token': token,
                'redirect': '/home',
                'email': user.email
            }), 200
        else:
            return jsonify({'error': 'Invalid email or password'}), 401

    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')
            

@app.route('/logout')
def logout():
    # Clear server-side session if it exists
    session.pop('user_id', None)
    return render_template('logout.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    """
    return render_template('chatbot.html') 
    """ 
    if request.method == 'GET':
        return jsonify({"message": "How Are You Feeling Today? 😊"})

    if request.method == 'POST':
        try:
            data = request.get_json()

            if not data:  # If no JSON is provided
                return jsonify({"error": "Invalid request format"}), 400
            
            user_input = data.get('message')
            follow_up_response = data.get('follow_up_message')

            if not user_input or user_input.strip() == "":
                return jsonify({"error": "Missing 'message' in request"}), 400

            if user_input is None or user_input.strip() == "":
                return jsonify({"Error":"messgae cannot be empty"}),400

            print(f"Received input: {user_input}")  

            user_id = 1  # Mock user_id, change as needed

        except Exception as e:
            return jsonify({"error": "Invalid request format"}), 200

        
        new_session = Session(user_id=user_id)
        db.session.add(new_session)
        db.session.commit()  

        try:
           
            analyzer = NRCAnalyser()
            sentiment_result = analyzer.generate_sentiment(user_input)
            
            print(f"Sentiment Result: {sentiment_result}")  

            dominant_emotion = sentiment_result.get('dominant_category', 'neutral')
            response = "I'm here to listen. How are you feeling today?"

            #default sentiment responses 

            if dominant_emotion in ["anxiety", "stress"]:
                response = "It sounds like you might be feeling anxious or stressed. It is always good to vent your emotions, if so respond with Talk, else I can offer you some relaxation techniques, type relaxation techniques for these"
            elif dominant_emotion == "joy":
                response = "I'm glad you're feeling happy! How can I help you today?"
            elif dominant_emotion == "anger":
                response = "I can sense you're feeling angry. Do you want to talk about what's bothering you, or maybe try some calming techniques?"
            elif dominant_emotion == "anticipation":
                response = "It seems you're feeling excited or looking forward to something. What's got you feeling this way?"
            elif dominant_emotion == "disgust":
                response = "It seems like something is upsetting you. Do you want to talk about it or maybe distract yourself?"
            elif dominant_emotion == "fear":
                response = "It sounds like you're feeling scared or anxious. Would you like to talk it through or find something calming to do?"
            elif dominant_emotion == "sadness":
                response = "I can tell you're feeling down. It's okay to feel sad, would you like to talk about it?"
            elif dominant_emotion == "surprise":
                response = "It seems something took you by surprise. Want to talk about what happened?"
            elif dominant_emotion == "trust":
                response = "It sounds like you're feeling confident or trusting. That's great! Is there anything specific you'd like to discuss?"
            elif dominant_emotion == "positive":
                response = "You're feeling positive! That's wonderful to hear. How can I help you continue this good feeling?"
            else:
                response = "I'm not sure how you're feeling, but I'm here to listen."

        #response and follow ups 

            if follow_up_response:
                print(f"follow-up message: {follow_up_response}")
                
                if "talk" in follow_up_response.lower():
                    print(f"Detected 'talk' follow-up for emotion: {dominant_emotion}")  # Debugging

                    if dominant_emotion == "anger":
                        response = "It's great to vent when angry! Are you feeling better?"
                    elif dominant_emotion == "fear":
                        response = "It's okay to feel scared sometimes. Talking helps, what's on your mind?"
                    elif dominant_emotion == "sadness":
                        response = "It's good to share your feelings when feeling sad. What's making you feel sad? Would you like to talk or would you like some resources?"
                    elif dominant_emotion == "anxiety" or dominant_emotion == "stress":
                        response = "Anxiety can be difficult to handle, its important to talk through your emotions or seek help in emergency situations. Feel free to voice your stresses but take a look at our resources on the home page or follow this link for support,The NHS (link)."
                    elif dominant_emotion == "joy":
                        response = "That's great! Anything specific you'd like to share?"
                    else:
                        response = "I'm here to talk. Let me know what's on your mind!"

            #handling techniques followups
            elif "techniques" in follow_up_response.lower():
                print(f"Detected 'techniques' follow-up for emotion: {detected_emotion}")

                if dominant_emotion == "anxiety" or dominant_emotion == "stress":
                        response = "Here are some relaxation techniques: deep breathing, meditation, and grounding exercises. Would you like more details?"
                else:
                        response = "I can share techniques for different situations. What would you like help with?" 
            
            new_analysis = SentimentAnalysis(
                session_id=new_session.id,
                user_input=user_input,
                result=dominant_emotion,
                method_used="NRCAnalyzer",
                confidence_score=sentiment_result.get('dominant_emotion_score', 0.0)
            )
            db.session.add(new_analysis)
            db.session.commit()

            return jsonify({"response": response, "sentiment": sentiment_result})

        except Exception as e:
            print(f"Error during sentiment analysis: {e}")  #debugging
            return jsonify({"response": "An error occurred while analyzing your sentiment.", "sentiment": {}})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
