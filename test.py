import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

# Load environment variables, typically used for configuration settings
load_dotenv()

def main():
    st.set_page_config(page_title="Category Classification", page_icon=":bird:")
    st.header("Category Classification :bird:")
    
    message = st.text_area("Customer message")

    if message:
        st.write("Generating category...")
        try:
            result = generate_response(message)
            st.info(result)
        except Exception as e:
            st.error(f"Error generating response: {e}")

loader = CSVLoader(file_path="data.csv")
documents = loader.load()

embeddings = HuggingFaceEmbeddings()

persist_directory = 'db'
vectordb = FAISS.from_documents(documents, embeddings)

def retrieve_info(query, k=3): 
    return vectordb.similarity_search(query, k=k)

prompt_template = """
You are an advanced AI specialized in customer service for an energy company. 
Your task is to analyze a customer's inquiry, classify it into the correct category, 
and find the most similar past inquiries along with their categories. 
This will help in providing a response that aligns with the company's best practices.

Customer Inquiry:
{message}

Similar Past Inquiries:
{best_practice}

Based on the analysis, give me the category this inquiry falls in.
Reply format: Category:"your answer"
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=prompt_template
)

def generate_response(message):
    similar_responses = retrieve_info(message)
    best_practice_text = "\n".join([doc.page_content for doc in similar_responses]) 
    data = {
        "model": "llama3:instruct",
        "prompt": prompt.format(message=message, best_practice=best_practice_text),
        "stream": False,
        "temperature": 0.5,
        "max_tokens": 50  
    }
    endpoint = "http://192.168.69.39:11434/api/generate" 
    response = requests.post(endpoint, json=data)

    # Error handling
    response.raise_for_status() 

    response_data = response.json()
    return response_data["response"]


if __name__ == '__main__':
    main()
