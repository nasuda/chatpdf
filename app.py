
import os
import streamlit as st 
import openai
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from langchain.llms import OpenAI

import os


def main():
    openai.api_key = st.secrets["api_key"]
    st.set_page_config(page_title="Chat with your PDF")
    st.header("Chat with your PDF")
    
    #upload pdf
    pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    # extract text from pdf
    text =''
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    #split text
    text_splitter = CharacterTextSplitter(
        separator ='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function =len
    )
    chunks = text_splitter.split_text(text)
    
    #create embeddings
    if len(chunks) > 0:
        embeddings = OpenAIEmbeddings()
        knowledge_base =FAISS.from_texts(chunks, embeddings)
        user_question=st.text_input("Ask your question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm,chain_type='stuff')
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)
    else:
        st.warning("No PDF file was uploaded. Please upload a valid PDF file.")
     

        
if __name__=="__main__":
    main()
    
