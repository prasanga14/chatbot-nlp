import os
import textwrap
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

FILE_PATH = "./documents/nepalDoc4.pdf"

# Create a loader
loader = PyPDFLoader(FILE_PATH)

# Split document into pages
document = loader.load_and_split()

# Preprocessing
def wrap_text_preserve_newlines(text, width=110):
    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_texts = '\n'.join(wrapped_lines)
    return wrapped_texts

# Text splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# print(docs[0])
# print(len(docs))

# Embedding
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Streamlit interface
st.title("NepalGeoGuide")

# query = 'What is best season to visit nepal'
query = st.text_input("Ask anything about Nepal's geography and tour:")

# doc = db.similarity_search(query)
# print(wrap_text_preserve_newlines(doc[0].page_content))

if query:
    doc = db.similarity_search(query)
    if doc:
        st.write("Answer:")
        st.write(wrap_text_preserve_newlines(doc[0].page_content))
    else:
        st.write("No relevant information found.")


