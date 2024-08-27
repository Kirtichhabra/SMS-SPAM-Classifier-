import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Load the vectorizer and model
tfidf = pickle.load(open(r'C:\Users\Testbook\Desktop\sms-spam-classfier\vectorizer4_pickle.pkl', 'rb'))
model = pickle.load(open(r'C:\Users\Testbook\Desktop\sms-spam-classfier\model4_pickle.pkl', 'rb'))

# Download necessary NLTK data
nltk.download('punkt', download_dir=r'C:\Users\Testbook\Desktop\sms-spam-classfier\nltk_data')
nltk.download('stopwords', download_dir=r'C:\Users\Testbook\Desktop\sms-spam-classfier\nltk_data')

ps = PorterStemmer()

# Function to preprocess and transform the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # Tokenize the text

    text = [word for word in text if word.isalnum()]  # Remove non-alphanumeric characters

    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]  # Remove stopwords and punctuation

    text = [ps.stem(word) for word in text]  # Stem the words
    return " ".join(text)

# Function to apply custom spam rules
def apply_custom_rules(text):
    spam_indicators = ['won', 'congratulations', 'prize', 'rupee', 'dollar']
    for word in spam_indicators:
        if word in text:
            return 1  # Flag as spam
    return 0  # Continue with the model's prediction

# Streamlit UI
st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)

    # Apply custom spam rules
    if apply_custom_rules(transformed_sms):
        st.header("Spam")
    else:
        # Continue with the model prediction
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
