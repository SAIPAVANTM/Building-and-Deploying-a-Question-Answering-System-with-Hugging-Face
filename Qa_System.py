"""
#########################################################################
   BUILDING AND DEPLOYING A QUESTION ANSWERING SYSTEM WITH HUGGING FACE
#########################################################################
"""

#----------------MODULES-USED-----------------------
import pickle
import re
from PIL import Image
import base64
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

#--------------LOAD-VECTOR-STORE----------------
def load_vectorstore(embeddings):
    return Chroma(persist_directory="vectorstore", embedding_function=embeddings)

#-----------------RAG-CHAIN-------------------------
def create_rag_chain(retriever, llm, prompt):
    return (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

#----------------------HOMEPAGE--------------------
def homepage():
    st.title("BUILDING AND DEPLOYING A QUESTION ANSWERING SYSTEM WITH HUGGING FACE")
    st.write("Project By Tumu Mani Sai Pavan")
    file = open("C:/Users/saipa/Downloads/aa.gif", "rb")
    contents = file.read()
    data_gif = base64.b64encode(contents).decode("utf-8")
    file.close()
    st.markdown(f'<img src="data:image/gif;base64,{data_gif}" style="width: 800px;" >',unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Overview")
    st.write("""Building and deploying a question-answering system with Hugging Face involves selecting an appropriate dataset, preprocessing the data for model readiness, and choosing a suitable model architecture. 
             The model is then fine-tuned on the selected dataset to optimize performance. 
             After fine-tuning, the model is evaluated using metrics like F1 score. 
             Finally, the system is deployed using platforms like Hugging Face Hub or other cloud services for real-time usage.""")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Skills Takeaway")
    st.write("""
                1) Dataset Selection
                2) Data Preprocessing
                3) Model Selection
                4) Model Fine-tuning
                5) Model Evaluation
                6) Model Deployment""")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("About me")
    st.write(""""Self-motivated computer science student with keen interest in coding". Engineer with a passion for machine learning, With a mix of academic knowledge, practical skills, and a growth-oriented attitude, I'm eager to make my debut in the AI & ML field.""")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Contact")
    st.write("For any queries or collaborations, feel free to reach out to me:")
    email_icon = Image.open("D:/Downloads/Email_image.jpg")
    st.write("Email:")
    col1, col2 = st.columns([0.4, 5])
    with col1:
        st.image(email_icon, width=50)
    with col2:
        st.write("tmsaipavan@gmail.com")
    lin_icon = Image.open("D:/Downloads/Linkedin_image.jpg")
    st.write("LinkedIn:")
    col1, col2 = st.columns([0.4, 5])
    with col1:
        st.image(lin_icon, width=50)
    with col2:
        st.write("[Sai Pavan TM](https://www.linkedin.com/in/saipavantm/)")

#-----------------QAPAGE-------------------------
def qapage():
    # Load the setup objects (LLM and prompt)
    with open("llm_prompt.pkl", "rb") as f:
        llm, prompt = pickle.load(f)

    # Create embeddings and reload the vector store
    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    # Create the RAG chain
    rag_chain = create_rag_chain(retriever, llm, prompt)
    
    # Interactive UI
    st.subheader("Ask Your Question")
    user_input = st.text_input("Input query:", key="query_input")
    
    if st.button("Submit"):
        if user_input:
            with st.spinner("Processing your query..."):
                result = rag_chain.invoke(user_input)
                f1 = rag_chain.invoke("whats the f1 score of the previous query, just give score dont need explanation")
                match = re.search(r'F1 score.*?(\d+\.\d+)', f1)
            
            # Display results
            st.markdown("### Response")
            st.markdown(f"<div class='main'><p>{result}</p></div>", unsafe_allow_html=True)
            
            if match:
                b = match.group(1)
                st.markdown(f"**<span style='color:blue'>F1 Score:</span> {b}**", unsafe_allow_html=True)
        else:
            st.warning("Please enter a query.")

#---------------=MAIN-FUNCTION-------------------
def main():
    app_mode = st.sidebar.radio("Go to", ("Homepage", "QA Part"))
    if app_mode == "Homepage":
        homepage()
    elif app_mode == "QA Part":
        qapage()

#---------------EXECUTE-MAIN-FUNCTION---------------
if __name__ == "__main__":
    main()
