from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

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


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response


app = FastAPI()

# Define a request model for FastAPI
class QueryModel(BaseModel):
    message: str

# Endpoint to receive the customer query and return the response
@app.post("/classify_query/")
async def classify_query(query: QueryModel):
    if not query.message:
        raise HTTPException(status_code=400, detail="No message provided")
    
    response = generate_response(query.message)
    return {"response": response}