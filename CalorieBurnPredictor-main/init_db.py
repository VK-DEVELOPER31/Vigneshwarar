from app import app, db
from models import User, Prediction
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

def init_db():
    # Create all tables
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
        
        # Add sample predictions for the last 7 days
        today = datetime.now()
        for i in range(7):
            date = today - timedelta(days=i)
            prediction = Prediction(
                user_id=test_user.id,
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
        print("Database initialized with test data!")

if __name__ == '__main__':
    init_db() 