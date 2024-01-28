import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables, typically used for configuration settings
load_dotenv()

# Initialize the Streamlit application
def main():
    st.set_page_config(page_title="Category classification", page_icon=":bird:")
    st.header("Category Classification :bird:")
    
    # Text area for user input (customer message)
    message = st.text_area("Customer message")

    # Process the input message when provided
    if message:
        st.write("Generating category...")
        result = generate_response(message)
        st.info(result)

# Load and vectorize the sales response data from a CSV file
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

# Generate embeddings for the documents using OpenAI's model
embeddings = OpenAIEmbeddings()
# Create a FAISS index for efficient similarity search of the document embeddings
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    """
    Performs a similarity search for the given query using the FAISS index.

    Args:
    query (str): The customer query string.

    Returns:
    list: A list of page contents from the top 3 similar historical queries.
    """
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Initialize the language model and prompt template for query categorization
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template="""
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
)
chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    """
    Generates a response for a given customer message by retrieving similar queries
    and using an AI model for categorization.

    Args:
    message (str): The customer query string.

    Returns:
    str: The categorized response.
    """
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

# Main function to launch the Streamlit app
if __name__ == '__main__':
    main()
