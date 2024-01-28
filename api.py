from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI application
app = FastAPI()

# Define a Pydantic model for the structure of the incoming customer query
class QueryModel(BaseModel):
    message: str

# Vectorize the sales response CSV data
# CSVLoader is used to load and process the CSV file containing historical queries
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

# OpenAIEmbeddings is used to transform text data into vector embeddings
embeddings = OpenAIEmbeddings()

# FAISS is used for efficient similarity search in high-dimensional space
# Here we create a FAISS index from the document embeddings
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    """
    Performs a similarity search for a given query using the FAISS index.

    Args:
    query (str): The customer query string.

    Returns:
    list: A list containing page contents of the top similar historical queries.
    """
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Setup LLMChain & prompts for processing queries
# ChatOpenAI is a wrapper for OpenAI's GPT models, used here for query categorization
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

# Template for processing customer queries using a predefined format
template = """
You are an advanced AI specialized in customer service for an energy company. 
Your task is to analyze a customer's inquiry, classify it into the correct category, 
and find the most similar past inquiries along with their categories. 
This will help in providing a response that aligns with the company's best practices.

Below is a customer's inquiry:
{message}

Based on the analysis, the inquiry falls into the following category like these examples
{best_practice}

Using this information, give me the category this enquiry falls in.
The reply should be like Category:"your answer"
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    """
    Generates a categorized response for a given customer message.

    Args:
    message (str): The customer query string.

    Returns:
    str: The categorized response.
    """
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

@app.post("/classify_query/")
async def classify_query(query: QueryModel):
    """
    API endpoint for classifying customer queries. 
    It receives a customer query and returns its categorized response.

    Args:
    query (QueryModel): The customer query in QueryModel format.

    Returns:
    dict: A dictionary containing the categorized response.
    """
    if not query.message:
        raise HTTPException(status_code=400, detail="No message provided")
    
    response = generate_response(query.message)
    return {"response": response}
