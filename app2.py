import os
import re
import string
import textwrap
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK dependencies are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

FILE_PATH = "./documents/nepalDoc4.pdf"

# Function to read PDF and split into pages
def read_pdf(file_path):
    pdf = PdfReader(open(file_path, 'rb'))
    text = []
    for page_num in range(len(pdf.pages)):
        text.append(pdf.pages[page_num].extract_text())
    return text

# Preprocessing function
def preprocess_text(text):
    # Remove unwanted characters, punctuation, and whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    
    # Tokenization
    tokens = word_tokenize(text)

    # print(tokens)
    
    # Stop Words Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Function to wrap text preserving new lines
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_texts = '\n'.join(wrapped_lines)
    return wrapped_texts

# Load and preprocess the document
document = read_pdf(FILE_PATH)
preprocessed_docs = [preprocess_text(page) for page in document]

# Vectorize the documents using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using bigrams for better context
doc_vectors = vectorizer.fit_transform(preprocessed_docs)

# Streamlit interface
st.title("NepalGeoGuide")

query = st.text_input("Ask anything about Nepal's geography and tour:")

if query:
    # Preprocess the query
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    
    # Calculate similarity
    similarities = cosine_similarity(query_vector, doc_vectors)
    most_similar_doc_index = similarities.argmax()
    
    st.write("Answer:")
    st.write(wrap_text_preserve_newlines(document[most_similar_doc_index]))
