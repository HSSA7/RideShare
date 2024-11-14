# app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///carpool.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # Retrieved from .env
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # Retrieved from .env

# Initialize Extensions
CORS(app, origins=["http://localhost:3000"])  # Adjust as per your frontend
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")  # Adjust as per your frontend

# Load ML model
try:
    model = joblib.load('model.pkl')
    print("ML model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading ML model: {e}")

# --- Database Models ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'Rider', 'Driver', or 'Both'

class RideRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rider_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pickup_location = db.Column(db.String(150), nullable=False)
    dropoff_location = db.Column(db.String(150), nullable=False)
    pickup_time = db.Column(db.DateTime, nullable=False)

class DrivingSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    driver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    route = db.Column(db.String(300), nullable=False)
    departure_time = db.Column(db.DateTime, nullable=False)
    available_seats = db.Column(db.Integer, nullable=False)

# Initialize the database
@app.before_first_request
def create_tables():
    db.create_all()

# --- Routes ---

# Home Route
@app.route('/', methods=['GET'])
def home():
    return jsonify(message="Final_Car Flask Backend is Running."), 200

# Register Route
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        # Validate incoming data
        required_fields = ["username", "password", "role"]
        if not data:
            return jsonify(message="No input data provided."), 400
        for field in required_fields:
            if field not in data:
                return jsonify(message=f"Missing field: {field}"), 400

        username = data['username'].strip().lower()
        password = data['password'].strip()
        role = data['role'].strip().capitalize()

        # Validate role
        if role not in ["Rider", "Driver", "Both"]:
            return jsonify(message="Invalid role. Choose from 'Rider', 'Driver', or 'Both'."), 400

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            return jsonify(message="Username already exists."), 409

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create new user
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()

        print("User registered successfully.")
        return jsonify(message="User registered successfully."), 201

    except Exception as e:
        print(f"Error in /register: {e}")
        return jsonify(message="Internal server error."), 500

# Login Route
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"Login data received: {data}")

        if not data or not all(k in data for k in ("username", "password")):
            return jsonify(message="Missing username or password."), 400

        username = data['username'].strip().lower()
        password = data['password'].strip()

        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify(message="Invalid credentials."), 401

        if not bcrypt.check_password_hash(user.password, password):
            return jsonify(message="Invalid credentials."), 401

        # Create JWT token (expires in 1 hour)
        access_token = create_access_token(
            identity={'user_id': user.id, 'role': user.role},
            expires_delta=timedelta(hours=1)
        )

        print(f"Access token generated for user '{user.username}'.")
        return jsonify(access_token=access_token), 200

    except Exception as e:
        print(f"Error during login: {e}")
        return jsonify(message="Internal server error."), 500

# Create Ride Request Route
@app.route('/ride-request', methods=['POST'])
@jwt_required()
def create_ride_request():
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        print(f"Ride request data received: {data}")

        # Validate incoming data
        required_fields = ["pickup_location", "dropoff_location", "pickup_time"]
        if not data:
            return jsonify(message="No input data provided."), 400
        for field in required_fields:
            if field not in data:
                return jsonify(message=f"Missing field: {field}"), 400

        pickup_location = data['pickup_location'].strip()
        dropoff_location = data['dropoff_location'].strip()
        try:
            pickup_time = datetime.fromisoformat(data['pickup_time'])
        except ValueError:
            return jsonify(message="Invalid pickup_time format. Use ISO 8601 format."), 400

        # Create new ride request
        new_request = RideRequest(
            rider_id=current_user['user_id'],
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            pickup_time=pickup_time
        )
        db.session.add(new_request)
        db.session.commit()

        # Emit SocketIO event
        socketio.emit('new_ride_request', {
            'message': f"New ride request from user ID {current_user['user_id']}."
        })

        #, removed Broadcast= true

        print("Ride request created successfully.")
        return jsonify(message="Ride request created successfully."), 201

    except Exception as e:
        print(f"Error in /ride-request: {e}")
        return jsonify(message="Internal server error."), 500

# Create Driving Schedule Route
@app.route('/driving-schedule', methods=['POST'])
@jwt_required()
def create_driving_schedule():
    try:
        current_user = get_jwt_identity()
        data = request.get_json()
        print(f"Driving schedule data received: {data}")

        # Validate incoming data
        required_fields = ["route", "departure_time", "available_seats"]
        if not data:
            return jsonify(message="No input data provided."), 400
        for field in required_fields:
            if field not in data:
                return jsonify(message=f"Missing field: {field}"), 400

        route = data['route'].strip()
        try:
            departure_time = datetime.fromisoformat(data['departure_time'])
        except ValueError:
            return jsonify(message="Invalid departure_time format. Use ISO 8601 format."), 400

        try:
            available_seats = int(data['available_seats'])
            if available_seats < 1:
                raise ValueError
        except ValueError:
            return jsonify(message="available_seats must be a positive integer."), 400

        # Create new driving schedule
        new_schedule = DrivingSchedule(
            driver_id=current_user['user_id'],
            route=route,
            departure_time=departure_time,
            available_seats=available_seats
        )
        db.session.add(new_schedule)
        db.session.commit()

        # Emit SocketIO event
        socketio.emit('new_driving_schedule', {
            'message': f"New driving schedule from user ID {current_user['user_id']}."
        })

        print("Driving schedule created successfully.")
        return jsonify(message="Driving schedule created successfully."), 201

    except Exception as e:
        print(f"Error in /driving-schedule: {e}")
        return jsonify(message="Internal server error."), 500

# Predict Route Zones
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_route_zones():
    try:
        data = request.get_json()
        print(f"Prediction data received: {data}")

        features = data.get('features')
        required_features = [
            'pickup_area',
            'dropoff_area',
            'day_of_week',
            'traffic_level',
            'weather_conditions',
            'available_seats',
            'historical_rides',
            'hour_of_day',
            'month'
        ]

        if not features:
            return jsonify(message="No features provided."), 400

        missing_features = [feat for feat in required_features if feat not in features]
        if missing_features:
            return jsonify(message=f"Missing features: {', '.join(missing_features)}"), 400

        # Prepare input DataFrame for prediction
        input_data = {key: [value] for key, value in features.items()}
        input_df = pd.DataFrame(input_data)

        # Ensure model is loaded
        if not model:
            return jsonify(message="ML model is not loaded."), 500

        # Make prediction
        prediction = model.predict(input_df)[0]
        print(float(prediction))
        prediction_probability = float(prediction)  # Assuming the model returns probabilities

        print(f"Prediction: {prediction_probability}")
        return jsonify(prediction=prediction_probability), 200

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify(message="Error processing prediction."), 500

# --- Socket.IO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

# --- Run the Application ---

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Default to 5001 if PORT not set
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
