"""
###############################################
   CREATING LLM FOR QUESTION ANSWERING SYSTEM
###############################################
"""


#--------------------MODULES-USED---------------------
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
import pickle

#------------------DATASET-LOADING-------------------
def load_documents():
    loader = PyPDFDirectoryLoader("D:/Downloads/GenAI")
    return loader.load()

#------------------CHUNKING-PROCESS------------------
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.split_documents(docs)

#------------------EMBEDDINGS-------------------------
def create_embeddings():
    return SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

#------------------VECTOR-STORE-----------------------
def create_vectorstore(chunks, embeddings):
    return Chroma.from_documents(chunks, embeddings, persist_directory="vectorstore")

#---------------------LLM------------------------------
def initialize_llm():
    return LlamaCpp(
        model_path="D:/Downloads/BioMistral-7B.Q4_K_M.gguf",
        temperature=0.2,
        max_tokens=2048,
        top_p=1
    )

#-------------------PROMPT-TEMPLATE--------------------
def create_prompt_template():
    template = """
<|context|>
You are a Medical Assistant that follows the instructions and generates accurate responses based on the query and the context provided.
Please be truthful and give direct answers.
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
    return ChatPromptTemplate.from_template(template)

#-----------------MAIN-FUNCTION---------------------
def main():
    # Set environment variables
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
    
    # Load and process documents
    docs = load_documents()
    chunks = split_documents(docs)
    
    # Create embeddings and vector store
    embeddings = create_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    
    # Persist the vector store to disk
    vectorstore.persist()

    # Initialize LLM and create prompt template
    llm = initialize_llm()
    prompt = create_prompt_template()

    # Save the LLM and prompt template only
    with open("llm_prompt.pkl", "wb") as f:
        pickle.dump((llm, prompt), f)

    print("Setup completed and saved!")


#--------------EXECUTE-MAIN-FUNCTION-------------
if __name__ == "__main__":
    main()
