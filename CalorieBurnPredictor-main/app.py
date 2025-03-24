from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pandas as pd
import joblib
import numpy as np
import logging
import traceback
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta, UTC
from models import db, Prediction, User
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from functools import wraps
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a secure secret key

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Create database tables and initialize test data
with app.app_context():
    db.create_all()
    
    # Check if test user exists
    test_user = User.query.filter_by(email='test@example.com').first()
    if not test_user:
        # Create test user
        test_user = User(
            name='Test User',
            email='test@example.com',
            password=generate_password_hash('password123')
        )
        db.session.add(test_user)
        db.session.commit()
    
    # Add sample predictions for all users if they don't have any
    today = datetime.now(UTC)
    all_users = User.query.all()
    
    for user in all_users:
        # Check if user has any predictions
        user_predictions = Prediction.query.filter_by(user_id=user.id).first()
        if not user_predictions:
            # Add sample predictions for the last 7 days
            for i in range(7):
                date = today - timedelta(days=i)
                prediction = Prediction(
                    user_id=user.id,
                    gender='Male' if i % 2 == 0 else 'Female',
                    height=175 + i,
                    weight=70 + i,
                    duration=30 + i * 5,
                    heart_rate=120 + i * 5,
                    calories_burned=250 + i * 50,
                    date=date
                )
                db.session.add(prediction)
    
    db.session.commit()
    logger.info("Database initialized with sample data for all users!")

# Define the exact feature order used in training
FEATURE_ORDER = ['Gender', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Weight_Duration_Interaction']

# Load the model
try:
    model = joblib.load('calories_tuned_model.pkl')
    logger.info("Model loaded successfully")
    if hasattr(model, 'feature_names_in_'):
        logger.info(f"Model expects features: {model.feature_names_in_}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid email or password'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered'})
        
        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            return jsonify({'success': True})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)})
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'user_id' not in session:
            raise ValueError("User not authenticated")
            
        # Get data from the form
        data = request.get_json()
        logger.debug(f"Received data: {data}")
        
        # Validate input data
        if not all(key in data for key in ['gender', 'height', 'weight', 'duration', 'heart_rate', 'date']):
            raise ValueError("Missing required input fields")
        
        # Convert and validate numeric inputs
        try:
            height = float(data['height'])
            weight = float(data['weight'])
            duration = float(data['duration'])
            heart_rate = float(data['heart_rate'])
            date = datetime.strptime(data['date'], '%Y-%m-%dT%H:%M')
        except ValueError as e:
            raise ValueError(f"Invalid numeric input: {str(e)}")
        
        # Create input DataFrame with correct feature order
        input_data = pd.DataFrame({
            'Gender': [1 if data['gender'].lower() == 'female' else 0],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Weight_Duration_Interaction': [weight * duration]
        })
        
        # Ensure all columns are float type and in correct order
        input_data = input_data[FEATURE_ORDER].astype(float)
        
        # Log the input data for debugging
        logger.debug(f"Input data for prediction: {input_data}")
        logger.debug(f"Input data types: {input_data.dtypes}")
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            logger.info(f"Prediction successful: {prediction}")
            
            # Store prediction in database with date and user_id
            new_prediction = Prediction(
                user_id=session['user_id'],  # Add user_id from session
                gender=data['gender'],
                height=float(data['height']),
                weight=float(data['weight']),
                duration=float(data['duration']),
                heart_rate=float(data['heart_rate']),
                calories_burned=float(prediction),
                date=date
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            # Generate graph
            plt.figure(figsize=(10, 6))
            features = ['Height', 'Weight', 'Duration', 'Heart_Rate']
            values = [float(data['height']), float(data['weight']), 
                     float(data['duration']), float(data['heart_rate'])]
            
            plt.bar(features, values)
            plt.title('Input Features Analysis')
            plt.ylabel('Value')
            plt.xticks(rotation=45)
            
            # Save plot to base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'plot': plot_url
            })
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Model prediction traceback: {traceback.format_exc()}")
            raise ValueError(f"Error during prediction: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    try:
        # Get predictions for the last 7 days for the current user
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=7)
        
        predictions = Prediction.query.filter(
            Prediction.user_id == session['user_id'],
            Prediction.date.between(start_date, end_date)
        ).order_by(Prediction.date).all()
        
        # Create interactive line graph using Plotly
        fig = go.Figure()
        
        if predictions:
            # If there are predictions, use real data
            dates = [p.date.strftime('%Y-%m-%d') for p in predictions]
            calories = [p.calories_burned for p in predictions]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=calories,
                mode='lines+markers',
                name='Calories Burned',
                line=dict(color='#4CAF50', width=2),
                marker=dict(size=8, color='#4CAF50'),
                hovertemplate="<b>Date:</b> %{x}<br>" +
                             "<b>Calories Burned:</b> %{y:.0f}<br>" +
                             "<extra></extra>"
            ))
        else:
            # If no predictions, show sample data
            dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6, -1, -1)]
            sample_calories = [250, 300, 275, 350, 325, 400, 375]  # Sample data
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=sample_calories,
                mode='lines+markers',
                name='Sample Calorie Burn',
                line=dict(color='#4CAF50', width=2, dash='dash'),
                marker=dict(size=8, color='#4CAF50'),
                hovertemplate="<b>Date:</b> %{x}<br>" +
                             "<b>Sample Calories Burned:</b> %{y:.0f}<br>" +
                             "<extra></extra>"
            ))
            
            # Add a note about sample data
            fig.add_annotation(
                text="This is sample data. Make your first prediction to see your actual calorie burn history!",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.95,
                showarrow=False,
                font=dict(size=12, color='#666')
            )
        
        # Update layout
        fig.update_layout(
            title='Daily Calorie Burn History',
            xaxis_title='Date',
            yaxis_title='Calories Burned',
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            tickangle=45
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey'
        )
        
        # Convert the figure to JSON for sending to the template
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'predictions': [p.to_dict() for p in predictions],
            'graph_json': graph_json
        })
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error fetching history: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True) 