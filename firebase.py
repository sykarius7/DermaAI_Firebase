import firebase_admin
from firebase_admin import credentials, auth

if not firebase_admin._apps:
    # Initialize the Firebase Admin SDK
    cred = credentials.Certificate("./serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

def create_user(email, password):
    user = auth.create_user(email=email, password=password)
    return user

def verify_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user
    except auth.AuthError as e:
        # Handle authentication error (e.g., invalid email or password)
        return None
