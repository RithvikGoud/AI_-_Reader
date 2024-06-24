import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pickle
import time

load_dotenv()
genai.configure(api_key=os.getenv("Google_api_key"))

def pdf_text(pdf_docs): 
    data=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            data+=page.extract_text()
    return data

def chunks(pdf_text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunk=text_splitter.split_text(pdf_text)
    return chunk

def get_vector_store(chunk):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(chunk,embedding=embeddings)
    save_directory="Store"
    vector_store.save_local(save_directory)
    return vector_store

def delete_data(save_dir: str,model: str="models/embedding-001"):
    embeddings = GoogleGenerativeAIEmbeddings(model=model)
    vectore_store=FAISS.load_local(save_dir,embeddings=embeddings,allow_dangerous_deserialization=True)
    vectore_store.index.reset()
    vectore_store.save_local(save_dir)
    file_path=r"C:\Users\ganesh\Desktop\LangChain\Store\index.pkl" # Run the application and upload pdf,then paste the path of index.pkl file from your pc.
    empty_data={}
    with open(file_path,"wb") as f:
        pickle.dump(empty_data,f)
    d=st.empty()
    d.warning("Deleting data......")
    time.sleep(5)
    d.empty()

def conversation_chain():
    prompt_template = """
            Please answer the following question based on the provided context.
            Your answer should be as detailed as possible, reflecting the information in the context.
            If the answer is not available in the context, respond with, "The answer is not available in the context."
            Context:\n {context}?\n
            Question:\n {question}\n

            Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.4)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.load_local("Store",embeddings,allow_dangerous_deserialization=True)
    documents = vector_store.similarity_search(user_question)
    chain = conversation_chain()
    response = chain(
        {"input_documents": documents, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("AI 👀 Reader")
    st.header("Spill the tea, Gemini💁!  What secrets are hiding in this PDF? ☕️")
    user_question = st.text_input("Ask a Question❓")
    with st.sidebar:
        st.title("Upload PDF's 📑 :")
        pdf_docs = st.file_uploader("Upload Your PDFs and Let's Get Processing! 🔥", accept_multiple_files=True)
        submit_button=st.button("Submit & Process")
        delete=st.button("Delete")
        if delete:
            if not pdf_docs:
                a=st.empty()
                a.warning("No files to delete")
                time.sleep(5)
                a.empty()
            else:
                try:
                    with st.spinner("Processing...."):
                        delete_data("Store")
                    a=st.empty()
                    a.success("Done")
                    time.sleep(3)
                    a.empty()
                except ValueError as er:
                    ee=st.empty()
                    ee.warning(f"No files to delete")
                    time.sleep(5)
                    ee.empty()
                except FileNotFoundError:
                    ee=st.empty()
                    ee.warning(f"FAISS index file not found or Already deleted!!!!")
                    time.sleep(5)
                    ee.empty()
                except PermissionError:
                    print(f"You don't have permission to delete.")
        elif submit_button:
            if not pdf_docs:
                a=st.empty()
                a.warning("Please Upload Files First.")
                time.sleep(5)
                a.empty()
            else:
                with st.spinner("Processing..."):
                    raw_text = pdf_text(pdf_docs)
                    text_chunks = chunks(raw_text)
                    get_vector_store(text_chunks)
                    b=st.empty()
                    b.warning("Uploading Files.......")
                    time.sleep(5)
                    b.empty()
                a=st.empty()
                a.success("Done")
                time.sleep(3)
                a.empty()

    if st.button("Ask?"):
        if user_question:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            try:
                vector_store = FAISS.load_local("Store", embeddings=embeddings, allow_dangerous_deserialization=True)
                if vector_store is not None:
                    user_input(user_question)
            except RuntimeError as e:
                a=st.empty()
                a.error("Upload the PDF before asking question❓")
                time.sleep(5)
                a.empty()
            except ValueError as e:
                a=st.empty()
                a.error("""File found empty!!!
                            Please Upload PDF's before asking question...""")
                time.sleep(5)
                a.empty()
        else:
            a=st.empty()
            a.warning("Please enter a question before asking.")
            time.sleep(5)
            a.empty()
    
if __name__ == "__main__":
    main()
