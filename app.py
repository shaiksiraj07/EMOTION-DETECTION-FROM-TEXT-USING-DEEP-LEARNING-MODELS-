import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Load Model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

def main():
    # ... all your imports remain unchanged

    # CSS updated in st.markdown:
    st.markdown("""
    <style>
        # /* Global Styles */
        # html, body, [class*="css"] {
        #     # background: linear-gradient(60deg, cornflowerblue, plum);
        #     # color: #000000 !important;
        #     font-family: 'Montserrat', sans-serif;
        # }

        .block-container {
            margin-top: 7rem;
            padding: 2.5rem;
            max-width: 900px;
            background: linear-gradient(135deg, rgba(200, 230, 255, 0.75), rgba(255, 220, 240, 0.75));
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(180, 200, 255, 0.3);
            margin-left: auto;
            margin-right: auto;
        }

        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Montserrat:wght@400;600&display=swap');

        .centered-title {
            font-style:italic;
            text-align: center;
            font-size: 52px;
            font-weight: 700;
            font-family: 'Poppins', sans-serif;
            margin-bottom: 15px;
            color: #C71585 ;
            text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.4);
            animation: fadeInUp 1.5s ease;
        }

        .subtitle {
            font-style:italic;
            text-align: center;
            font-size: 22px;
            font-family: 'Montserrat', sans-serif;
            max-width: 700px;
            margin: 0 auto 2rem auto;
            color: #000000 !important;
            animation: fadeInUp 1.8s ease;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, rgba(255, 255, 204, 0.75), rgba(204, 255, 229, 0.75));
            margin-top: 2rem;
            padding: 1rem;
            margin-left: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(5px);
            color: #000000 !important;
        }

        [data-testid="stSidebar"] .sidebar-content {
            background: transparent !important;
            color: #000000 !important;
         }

        /* ✅ UPDATED: Select Menu Styles - Light background, black text */
        [data-testid="stSidebar"] select {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            color: #000000 !important;
            border: 1px solid rgba(0, 0, 0, 0.2);
        }

        [data-testid="stSidebar"] option {
            color: #000000 !important;
            background-color: white;
        }

        /* Form & Text Area Styles */
        .stTextArea textarea {
            border-radius: 12px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            padding: 1rem;
            font-size: 16px;
            background-color:#FFEBCD;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            color:black;
            font-weight:1000;

            
        }

        .stTextArea textarea:focus {
            border-color: #000000;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            background-color: #FFDEAD	;
            color:black;
            font-weight:1000;
        }

        .stButton>button {
            background-color: white;
            color: #ffffff !important;
            border-radius: 8px;
            padding: 0.7rem 1.5rem;
            font-weight: 1000;
            font-family: 'Montserrat', sans-serif;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            background: linear-gradient(to right, #2575fc, #6a11cb);
           
        }

        .stSubheader {
            font-family: 'Poppins', sans-serif;
            font-size: 1.8rem;
            color: #000000 !important;
            text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.4);
            margin-top: 2rem;
            animation: fadeInUp 1s ease;
        }

        .stColumn {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
            animation: fadeInUp 1.2s ease;
            color: #000000 !important;
        }

        .stAlert {
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(5px);
        }

        .stAlert [data-testid="stMarkdownContainer"] {
            color: #000000 !important;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        .prediction-result {
            font-size: 2rem;
            font-weight: 1000;
            color:black;
           
                
        }

        .emoji {
            font-size: 2rem;
            vertical-align: middle;
        }

        .stExpander {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
        }

        .stExpander .streamlit-expanderHeader {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            color: #000000 !important;
            background-color: transparent;
        }

        .stDataFrame {
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.1);
            color: #000000 !important;
        }

        .about-section {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(5px);
            animation: fadeInUp 1s ease;
            color: #000000 !important;
            
        }

        .about-section h3 {
            color: #000000 !important;
            font-family: 'Poppins', sans-serif;
            text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.4);
          
        }

        .about-section ul {
            padding-left: 1.5rem;
        }

        .about-section li {
            margin-bottom: 0.5rem;
            color: #000000 !important;
        }

        @keyframes fadeInUp {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        div{
                color:#FF4500;
                font-style:bold;
                font-weight:1000;
                font-size:1rem;
                }
        
        h1, h2, h3, h4, h5, h6 {
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)


    # Display the centered title and stylish subtitle with animations
    st.markdown("<div class='centered-title'>EmotionClassifier</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>\"See how your words really feel\"</div>", unsafe_allow_html=True)

    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here", height=200)
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))

        st.write("Welcome to the EmotionClassifier! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")
        st.write("At EmotionClassifier, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text...")

        st.subheader("How It Works")
        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text...")

        st.subheader("Key Features:")
        st.markdown("##### 1. Real-time Emotion Detection")
        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text...")

        st.markdown("##### 2. Confidence Score")
        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions...")

        st.markdown("##### 3. User-friendly Interface")
        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text...")

        st.subheader("Applications")
        st.markdown("""
          The EmotionClassifier App has a wide range of applications across various industries and domains. Some common use cases include:
          - Social media sentiment analysis
          - Customer feedback analysis
          - Market research and consumer insights
          - Brand monitoring and reputation management
          - Content analysis and recommendation systems
        """)

if __name__ == '__main__':
    main()