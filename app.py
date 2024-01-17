import streamlit as st
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import plotly.express as px
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
from streamlit.components.v1 import html
import pickle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# Streamlit settings
st.set_page_config(page_title="FER-2013", layout="wide")
st.markdown(
    """
    <div class="result-card" style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"> <h1 style='text-align: center; color: #11111; '>🔔 Facial Emotion Recognition</h1></div>
    """,
    unsafe_allow_html=True
)
st.markdown(
        """
        <head>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        </head>
        """,
        unsafe_allow_html=True
)
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Dictionary to map class indices to emotions
class_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
# Load the pre-trained model
model = tf.keras.models.load_model('model3.h5')

# Load the information to recreate the generator
with open('test_generator_info.pkl', 'rb') as file:
    test_generator_info = pickle.load(file)

# Function to recreate the generator
def recreate_test_generator():
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2,
    )
    return datagen.flow_from_directory(**test_generator_info)
def generate_confusion_matrix(model, test_generator):
    # Get predictions on the test set
    y_pred = model.predict(test_generator)
    # Get true labels from the test generator
    y_true = test_generator.labels
    # Generate a confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred.argmax(axis=1))
    return conf_matrix



# Placeholder for model insights data
model_insights_data = {
    "Model Architecture": ["CNN"],
    "Training Dataset": ["FER 2013 (7178 images)"],
    "Optimizer": ["Adam (learning rate = 0.0001)"],
    "Loss Function": ["Categorical Crossentropy"],
    "Metrics": ["Categorical Accuracy"],
    "Augmentation": ["Image augmentation with shear, zoom, and horizontal flip"],
    "Training Epochs": [20],
}

# Create a DataFrame for model insights
model_insights_df = pd.DataFrame(model_insights_data)

# Placeholder for insights table
insights_data = {
    "Tested Model": ["Model 1", "Model 2", "Model 3"],
    "Accuracy": [0.5287, 0.499, 0.5478],
    "Loss": [1.3721, 1.3396, 1.6306],
}
insights_df = pd.DataFrame(insights_data)
# Additional insights data
additional_insights_data = {
    "Tested Model": ["Model 1", "Model 2", "Model 3"],
    "Precision": [0.534, 0.538, 0.525],
    "Recall": [0.4742, 0.4185, 0.5028],
    "F1 Score": [0.49, 0.4257, 0.510],
}

# Create a DataFrame for additional insights
additional_insights_df = pd.DataFrame(additional_insights_data)

# Function to display additional insights chart
def display_additional_insights_chart():
    fig = go.Figure()

    fig.add_trace(go.Bar(x=additional_insights_df['Tested Model'], y=additional_insights_df['Precision'], name='Precision'))
    fig.add_trace(go.Bar(x=additional_insights_df['Tested Model'], y=additional_insights_df['Recall'], name='Recall'))
    fig.add_trace(go.Bar(x=additional_insights_df['Tested Model'], y=additional_insights_df['F1 Score'], name='F1 Score'))

    fig.update_layout(
        barmode='group',
        title='Precision , Recall Tests and F1 Score',
        xaxis_title='Tested Model',
        yaxis_title='Metric Value',
        legend=dict(x=0.85, y=1.0),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig)

# Function to preprocess an image before passing it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Function to display cover images
def display_emotion_images():
    emotions_images = {
        'Angry': 'D:\\workspace\\FER-project\\dataset\\test\\angry\\PrivateTest_88305.jpg',
        'Disgust': 'D:\\workspace\\FER-project\\dataset\\test\\disgust\\PrivateTest_3838250.jpg',
        'Fear': 'D:\\workspace\\FER-project\\dataset\\test\\fear\\PrivateTest_166793.jpg',
        'Happy': 'D:\\workspace\\FER-project\\dataset\\test\\happy\\PrivateTest_218533.jpg',
        'Neutral': 'D:\\workspace\\FER-project\\dataset\\test\\neutral\\PrivateTest_59059.jpg',
        'Sad': 'D:\\workspace\\FER-project\\dataset\\test\\sad\\PrivateTest_366361.jpg',
        'Surprise': 'D:\\workspace\\FER-project\\dataset\\test\\surprise\\PrivateTest_104142.jpg',
    }
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    for emotion, img_path in emotions_images.items():
        col = col1 if emotion == 'Angry' else col2 if emotion == 'Disgust' else col3 if emotion == 'Fear' else col4 if emotion == 'Happy' else col5 if emotion == 'Neutral' else col6 if emotion == 'Sad' else col7
        col.write(f'*{emotion}*')
        img = Image.open(img_path)
        col.image(img, caption=emotion, use_column_width=True)

# Function to display model details
def display_model_details():
    # Add FontAwesome CSS
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
        """, unsafe_allow_html=True
    )

    # Model Details Section
    st.markdown('<div style = "font-color:#0068C9;"><H2> Model Details:</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-details-container">
            <ul>
                <li><i class="fas fa-cogs"></i> <strong>Architecture:</strong> Convolutional Neural Network (CNN)</li>
                <li><i class="fas fa-database"></i> <strong>Training Dataset:</strong> FER 2013 (28709 images)</li>
                <li><i class="fas fa-database"></i> <strong>Test Dataset:</strong> FER 2013 (7178 images)</li>
                <li><i class="fas fa-cogs"></i> <strong>Convolutional layers :</strong> 3 layers (32 (3x3) , 64(5x5), 128(5x5) ) </li>
                <li><i class="fas fa-cogs"></i> <strong>Fully connected layers :</strong> 5 layers (1: 512 neural , 2 :256, 3:128, 4: 64, 5:32 ) </li>
                <li><i class="fas fa-sliders-h"></i> <strong>Training Parameters:</strong>
                    <ul class="sub-list">
                        <li><i class="fas fa-chart-line"></i> <em>Optimizer:</em>  Adam (learning rate = 0.0001)</li>
                        <li><i class="fas fa-chart-line"></i> <em>Loss Function: </em> Categorical Crossentropy</li>
                        <li><i class="fas fa-chart-bar"></i> <em>Metrics:</em>  Categorical Accuracy</li>
                    </ul>
                </li>
                <li><i class="fas fa-expand"></i> <strong>Augmentation: </strong> Image augmentation with shear, zoom, and horizontal flip</li>
                <li><i class="far fa-clock"></i> <strong>Training Epochs:  </strong> 20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

def display_insights_chart():
    fig = go.Figure()

    fig.add_trace(go.Bar(x=insights_df['Tested Model'], y=insights_df['Accuracy'], name='Accuracy'))
    fig.add_trace(go.Bar(x=insights_df['Tested Model'], y=insights_df['Loss'], name='Loss'))

    fig.update_layout(
        barmode='group',
        title='Accuracy Test',
        xaxis_title='Tested Model',
        yaxis_title='Metric Value',
        legend=dict(x=0.85, y=0.9),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig)
# Function to display model insights table
def display_insights_table():
    st.markdown('<div style = "font-color:#0068C9;"><H2> Tested Models Insights:</h2></div>', unsafe_allow_html=True)
    # Add a paragraph before the insights tables
    st.markdown('<p class="description">After testing various models, we present the insights from the best-performing models : </p>', unsafe_allow_html=True)
    # Use columns to display charts side by side
    col1, col2 = st.columns(2)
    
    # Display the first insights chart in the first column
    with col1:
        display_insights_chart()

    # Display the additional insights chart in the second column
    with col2:
        display_additional_insights_chart()


# Welcome Page
def welcome_page():
    st.markdown('<div class="emotion-images">', unsafe_allow_html=True)
    display_emotion_images()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="description"> Facial Emotion Recognition (FER) is a computer vision project that focuses on detecting and classifying human emotions from facial expressions. The goal is to build a model that can accurately identify emotions such as happiness, sadness, anger, surprise, fear, disgust, and neutrality based on facial images..</p>', unsafe_allow_html=True)

    display_model_details()
    # displaying ROC curve
    st.subheader('The Receiver Operating Characteristic (ROC) Curve')
    img_path="D:\\workspace\\FER-project\\assets\\c3b9c820e37e668f0657bc15f1904dad24815d2ddbefa5d1ef47b693.png"
    img = Image.open(img_path)
    st.image(img, caption='Receiver Operating Characteristic (ROC) Curve', use_column_width=True)


    display_insights_table()
    


def prediction_page(validation_loss, validation_accuracy,conf_matrix):
    
    display_emotion_images()
    # Create a two-column layout
    col_left, col_right = st.columns(2)

    # Column 1 (Left): Display the uploaded image (4x4 size) and Predicted Result
    with col_left:
        st.subheader('Emotion prediction section : ')
        uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Make the prediction when the 'Predict' button is clicked
            # Display the predicted result as a card
            if st.button('Predict'):
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_file)
                # Make the prediction
                predictions = model.predict(img_array)
                # Get the predicted emotion
                predicted_class = np.argmax(predictions)
                emotion = class_mapping[predicted_class]
                st.markdown('</div>', unsafe_allow_html=True)
                # Display the predicted result as a card
                st.markdown(f'<div class="result-card" style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);"> The result of the given image gives the predicted emotion <div class="result">{emotion}</div> with {predictions[0][predicted_class]*100:.2f}% probability.', unsafe_allow_html=True)
                # Display the chart with full column width
                fig_probs = px.bar(
                    x=list(class_mapping.values()),
                    y=predictions[0],
                    labels={'x': 'Emotion', 'y': 'Probability'},
                    title='Prediction Probabilities for Each Class',
                    color_discrete_sequence=['#ff2b2b'] * len(class_mapping),
                )

                st.plotly_chart(fig_probs, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # Column 2 (Right): Display the Validation Metrics and Confusion Matrix
    with col_right:
        # Create a Matplotlib figure and axis
        st.subheader('Confusion Matrix:')

        fig, ax = plt.subplots(figsize=(4, 2))
        # Display confusion matrix heatmap
        conf_matrix_heatmap = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

        # Display the Matplotlib figure using Streamlit
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)
        # Display validation accuracy
        st.subheader(f'Validation Accuracy: {validation_accuracy:.2%}')
        # Display validation loss
        st.subheader(f'Validation Loss: {validation_loss:.4f}')

def how_to_page():
    display_emotion_images()

    # Add icons to specific sections
  

    # Centered Header with Icon
    st.markdown('<div class="centered-header">', unsafe_allow_html=True)
    st.markdown('<h1> Facial Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Project Overview with Icon
    st.markdown('<div class="overview">', unsafe_allow_html=True)
    st.subheader('Project Overview:')
    st.write("""
        Facial Emotion Recognition (FER) is a computer vision project designed to detect and classify human emotions from facial expressions. The primary objectives include emotion classification, data preparation, model architecture design, training, and creating a user-friendly web interface using Streamlit.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # App Usage Overview with Icon
    st.markdown('<div class="overview">', unsafe_allow_html=True)
    st.subheader('App Usage Overview:')
    st.write("""
        The Facial Emotion Recognition app allows users to upload an image containing a human face and receive predictions about the person's emotion. Follow the steps below to use the app effectively:
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Steps to Use the App with Icon
    st.markdown('<div class="steps">', unsafe_allow_html=True)
    st.subheader(' Steps to Use the App:')
    st.write("""
        1. **Navigate to the 'Prediction' Section:**
           - Click on the 'Prediction' option in the sidebar menu to access the prediction page.
        2. **Upload an Image:**
           - Use the 'Choose an image...' button to upload an image from your device. Ensure that the image contains a clear view of a human face.
        3. **View Predictions:**
           - Once the image is uploaded, the app will display the uploaded image along with the predicted emotion. The predicted emotion will be one of the following: Angry, Disgust, Fear, Happy, Neutral, Sad, or Surprise.
        4. **Explore Model Details and Insights:**
           - Navigate to the 'Home' section in the sidebar to explore additional details about the model, including architecture, training parameters, and insights gained during the project.
        5. **Learn More:**
           - Explore the 'Home' section to discover more about the project, including its objectives, components, and technologies used.
        6. **Navigate Between Sections:**
           - Utilize the sidebar menu to switch between different sections of the app, including 'Home,' 'Prediction,' and 'How-to-do.'
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Additional Tips with Icon
    st.markdown('<div class="tips">', unsafe_allow_html=True)
    st.subheader('Additional Tips:')
    st.write("""
        - Ensure that the uploaded image contains a visible and well-lit human face for accurate emotion predictions.
        - Experiment with different images to observe how the model responds to various facial expressions.
        - Explore the 'Model Insights' section in the 'Home' page to gain insights into the model architecture and training process.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Objectives with Icon
    st.markdown('<div class="objectives">', unsafe_allow_html=True)
    st.subheader('Objectives:')
    st.write("""
        1. **Emotion Classification:**
           - Develop a Convolutional Neural Network (CNN) model capable of classifying facial expressions into predefined emotion categories.
        2. **Data Preparation:**
           - Utilize the FER 2013 dataset, consisting of 48x48-pixel grayscale images labeled with seven different emotions. Preprocess and augment the data for improved model training.
        3. **Model Architecture:**
           - Design a CNN architecture with layers for convolution, pooling, and fully connected networks to extract features from facial images and make emotion predictions.
        4. **Training and Optimization:**
           - Train the model using an appropriate optimizer, loss function, and metrics. Optimize the model parameters to achieve high accuracy in emotion prediction.
        5. **User Interface:**
           - Develop a Streamlit-based web application to provide an interactive and user-friendly interface for emotion prediction. Users can upload an image, and the model will predict the corresponding emotion.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Components with Icon
    st.markdown('<div class="components">', unsafe_allow_html=True)
    st.subheader('Components:')
    st.write("""
        1. **Model Training:**
           - Train the facial emotion recognition model using the FER 2013 dataset. Experiment with different architectures and hyperparameters to improve accuracy.
        2. **Data Augmentation:**
           - Apply data augmentation techniques such as shear, zoom, and horizontal flip to artificially increase the diversity of the training dataset.
        3. **Streamlit Web Application:**
           - Create a web application using Streamlit to allow users to interact with the trained model. Users can upload images, and the application will display the predicted emotion.
        4. **Model Insights:**
           - Display insights about the trained model, including architecture details, training dataset information, optimization parameters, and augmentation techniques.
        5. **Project Insights:**
           - Provide insights into the project, including the testing of multiple models before selecting the best one, accuracy, loss, and training time for each model.
        6. **User Guidance:**
           - Include guidance and instructions within the application on how to use it effectively for emotion prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Technologies Used with Icon
    st.markdown('<div class="technologies">', unsafe_allow_html=True)
    st.subheader('Technologies Used:')
    st.write("""
        - **Deep Learning Framework:** TensorFlow and Keras
        - **Web Application Framework:** Streamlit
        - **Data Visualization:** Plotly Express
        - **Image Processing:** PIL (Python Imaging Library)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Conclusion with Icon
    st.markdown('<div class="conclusion">', unsafe_allow_html=True)
    st.subheader('Conclusion:')
    st.write("""
        The Facial Emotion Recognition project aims to demonstrate the capabilities of deep learning in recognizing and classifying human emotions based on facial expressions. The web application provides an accessible platform for users to experience and understand the model's predictions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def sideBar():
    with st.sidebar:
        selected=option_menu(
            menu_title="Main Menu",
            options=["Home","Prediction","How-to-do"],
            icons=["house","eye","robot"],
            menu_icon="cast",
            default_index=0
        )
    if selected=="Home":
        #st.subheader(f"Page: {selected}")
        welcome_page()
    if selected=="Prediction":
        #st.subheader(f"Page: {selected}")
        test_generator = recreate_test_generator()
        validation_loss, validation_accuracy = model.evaluate(test_generator)
        conf_matrix = generate_confusion_matrix(model, test_generator)
        prediction_page(validation_loss, validation_accuracy,conf_matrix)
    if selected=="How-to-do":
        #st.subheader(f"Page: {selected}")
        how_to_page()

sideBar()