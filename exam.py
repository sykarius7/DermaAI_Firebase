import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
from firebase import  create_user, verify_user
from firebase_admin import  auth
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import google.generativeai as genai

st.set_page_config(page_title="Skin Disease Classification", page_icon=":microscope:", layout="wide")

with open( "./style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)



class_names = [
    'Acne and Rosacea Photos', 'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos', 'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles', 'No Disease','Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives',
    'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections'
]

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_image_class(model, image):
    image = np.array(image.resize((400, 400))) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    predicted_class = np.argmax(pred, axis=1)
    return class_names[predicted_class[0]]

model = genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def intro_page():
    st.markdown(
    "<h1 style='text-align:center;'>Welcome to DermAI!</h1>",
    unsafe_allow_html=True
)
    st.markdown(
        "<h2 style='text-align: center;'>This app helps you classify skin diseases and get answers about skin conditions.</h2>",
        unsafe_allow_html=True
    )

    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col2:
        if st.button("Sign Up"):
            st.session_state.page = "SignUp"
            st.rerun()

    with col4:
        if st.button("Login"):
            st.session_state.page = "Login"
            st.rerun()

def feedback_page():
    st.markdown(
    "<h1 style='text-align:center;'>Feedback</h1>",
    unsafe_allow_html=True
)
    st.markdown(
    "<h2 style='text-align:center;'> We value your feedback!</h2>",
    unsafe_allow_html=True
)
    feedback = st.text_area("Please provide your feedback here:")
    st.markdown("""
        <style>
        .stButton button {
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("Submit"):
        # Process the feedback (e.g., save it to a database or send it via email)
        if feedback:
            st.success("Thank you for your feedback!")
            # Redirect to the intro page after submitting feedback
            st.session_state.page = "Intro"
            st.rerun()
        else:
            st.error("Please provide feedback")

def login_page():
    st.markdown(
    "<h1 style='text-align:center;'> Login</h1>",
    unsafe_allow_html=True
)
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("Login"):
            try:
                user = verify_user(email, password)
                st.success(f"Logged In as {email}")
                st.session_state.page = "Main"
                st.session_state.user = user
                st.rerun()
            except Exception as e:
                st.error(f"Invalid credentials or user not found: {e}")
    with col4:
        if st.button('Back'):
            st.session_state.page = "Intro"
            st.session_state.user=None
            st.rerun()

def signup_page():
    st.markdown(
    "<h1 style='text-align:center;'> SignUp</h1>",
    unsafe_allow_html=True
)
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col2:
        if st.button("Sign Up"):
            try:
                user = create_user(email, password)
                st.success("User created successfully")
                st.session_state.page = "Login"
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    with col4:
        if st.button('Back'):
            st.session_state.page = "Intro"
            st.session_state.user=None
            st.rerun()

def generate_pdf(image, classification_result, chat_history):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    # Add image to PDF
    image_path = "uploaded_image.png"
    image.save(image_path)
    c.drawImage(image_path, 100, 600, width=400, height=400)

    # Add classification result to PDF
    c.drawString(100, 580, f"Classification Result: {classification_result}")

    # Add chatbot history to PDF
    c.drawString(100, 550, "Chatbot History:")
    y_position = 530
    for role, text in chat_history:
        c.drawString(120, y_position, f"{role}: {text}")
        y_position -= 20

    c.save()

    # Save PDF to a file
    buffer.seek(0)
    pdf_bytes = buffer.getvalue()
    return pdf_bytes



def main_page():
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Go to", ["Home", "DermAI", "ChatBot","Exit"])

    if option == "Home":
        st.write("## Welcome to the Skin Disease Classification App!")

        st.write("### Introduction")
        st.write("Welcome to our Skin Disease Classification app, where cutting-edge technology meets dermatology. Our app utilizes state-of-the-art machine learning algorithms to analyze images of skin conditions and provide accurate classification results.")

        st.write("### How to Use")
        st.write("Using our app is simple and intuitive. Just upload an image of the skin condition you want to analyze, and our algorithm will swiftly process it, providing you with detailed information about the detected disease.")

        st.write("### Key Features")
        st.write("- **Accurate Classification**: Our app employs advanced machine learning techniques to ensure precise identification of various skin diseases.")
        st.write("- **User-Friendly Interface**: Navigate through the app effortlessly with our intuitive and easy-to-use interface.")
        st.write("- **Fast Processing**: Get instant results with our speedy processing capabilities, allowing you to take prompt action.")
        st.write("- **Cross-Platform Compatibility**: Access the app from anywhere, on any device, ensuring convenience and flexibility.")

        st.write("### Benefits")
        st.write("- **Early Detection**: Detect potential skin issues early, empowering you to seek timely medical advice and treatment.")
        st.write("- **Track Progress**: Monitor changes in your skin health over time and track the effectiveness of treatment.")
        st.write("- **Convenience**: Access dermatological insights from the comfort of your own home, saving time and hassle.")

        st.write("### Data Privacy and Security")
        st.write("We take your privacy and security seriously. Rest assured that all data and images uploaded to our app are treated with the utmost confidentiality and are securely stored.")

        st.write("### Support and Contact Information")
        st.write("For any questions, concerns, or technical assistance, please reach out to our support team at support@skinclassificationapp.com.")

        st.write("### Future Developments")
        st.write("Stay tuned for exciting updates and new features coming soon to the Skin Disease Classification app!")

        st.write("Thank you for choosing our app to assist you in your skin health journey. Let's get started!")


    elif option == "DermAI":
        st.markdown(
    "<h1 style='text-align:center;'>Skin Disease Classification </h1>",
    unsafe_allow_html=True
)
        st.markdown(
    "<h5 style='text-align:center;'>This app is a tool for predicting skin disease types. Upload an image to see the predicted class. Please note that this is a tool for educational purposes and the prediction is not a diagnosis. </h5>",
    unsafe_allow_html=True
)
        
        uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
        st.markdown(
    "<h2 style='text-align:center;'>OR </h2>",
    unsafe_allow_html=True
)
        camera_file = st.camera_input("Take a photo")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write('Classifying...')
            model = load_model('bestmodel1.h5')
            predicted_class = predict_image_class(model, image)
            st.success(f'Predicted Class: {predicted_class}')
            st.session_state['uploaded_image'] = image
            st.session_state['predicted_class'] = predicted_class

        if camera_file is not None:
            image1 = Image.open(camera_file)
            st.success("Photo uploaded successfully")
            st.image(image1, caption='Uploaded Image', use_column_width=True)
            st.write('Classifying...')
            model = load_model('bestmodel1.h5')
            predicted_class = predict_image_class(model, image1)
            st.success(f'Predicted Class: {predicted_class}')
            st.session_state['uploaded_image'] = image1
            st.session_state['predicted_class'] = predicted_class
        st.markdown("""
        <style>
        .stButton button {
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)
        
        generate_report=st.button("Generate_Report")
        if generate_report:
            if 'uploaded_image' in st.session_state and 'predicted_class' in st.session_state:
                image = st.session_state['uploaded_image']
                classification_result = st.session_state['predicted_class']
                user_chat_history_key = f"{st.session_state.user}_chat_history"
                chat_history = st.session_state.get(user_chat_history_key, [])
                pdf_bytes = generate_pdf(image, classification_result, chat_history)
                st.download_button("Download PDF", data=pdf_bytes, file_name="classification_report.pdf", mime="application/pdf")
               
            else:
            
                st.error("Please upload an image or take a photo to generate a report.")

    elif option == "ChatBot":
        st.markdown(
    "<h1 style='text-align:center;'>Ask DermAI anything about skin conditions!</h1>",
    unsafe_allow_html=True
)
        input_text = st.text_input("Input: ", key="input")
        st.markdown("""
        <style>
        .stButton button {
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)
        submit_button = st.button("Ask the question")
        user_chat_history_key = f"{st.session_state.user}_chat_history"
        chat_history = st.session_state.get(user_chat_history_key, [])
        if submit_button and input_text:
            response = get_gemini_response(input_text)
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)

            # Update the chat history with the current input and response
            chat_history.append(("You", input_text))
            for chunk in response:
                chat_history.append(("Bot", chunk.text))
            st.session_state[user_chat_history_key] = chat_history

        if chat_history:
            st.markdown('<h3 style="color: yellow;">The Chat History is</h3>', unsafe_allow_html=True)
            for role, text in chat_history:
                if role == "You":
                    st.markdown(f'<span style="color: blue;">You: {text}</span>', unsafe_allow_html=True)
                elif role == "Bot":
                    st.markdown(f'<span style="color: green;">Bot: {text}</span>', unsafe_allow_html=True)

    elif option=='Exit':
        auth.current_user = None
        st.session_state.page = "Feedback"
        st.session_state.user=None
        feedback_page()
        st.rerun()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "Intro"

    if st.session_state.page == "Intro":
        intro_page()
    elif st.session_state.page == "Login":
        login_page()
    elif st.session_state.page == "SignUp":
        signup_page()
    elif st.session_state.page == "Main":
        main_page()
    elif st.session_state.page=="Feedback":
        feedback_page()

if __name__ == '__main__':
    main()
