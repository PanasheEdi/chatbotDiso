import requests
import json

def send_message(message):
    url = "http://127.0.0.1:5000/chat" 
    headers = {
        'Content-Type': 'application/json',
    }
    