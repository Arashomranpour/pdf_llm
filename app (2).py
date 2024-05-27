import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate, prompt
import os


genai.configure(api_key="AIzaSyDaFygGK9ocbwn1JRNKrB5_4H59dXmd8Dg")

def get_text(pdf_doc):
  text=""
  for pdf in pdf_doc:
    pdf_reader=PdfReader(pdf)
    for page in pdf_reader.pages:
      text+=page.extract_text()
  return text


def get_chunks(text):
  text_spliter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
  chunks=text_spliter.split_text(text)
  return chunks

def get_vect(chunk):
  embd=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyDaFygGK9ocbwn1JRNKrB5_4H59dXmd8Dg")
  vect=FAISS.from_texts(chunk,embd)
  vect.save_local("faiss_index")

def get_conv():
  prompt_template="""
  answer to questions as detailed as possible based on the provided context,
  make sure to provide all the details,
  if answer was not in context just say
  "not in context" and don't provide the wrong answer
  Context:\n {context}?\n
  Question :\n {question}\n
  Answer :

  """
  model=ChatGoogleGenerativeAI(model="Gemini-pro")
  p=PromptTemplate(prompt_template,input_variables=["content","question"])
  chain=load_qa_chain(model,chain_type="stuff",prompt=p)
  return chain

def user_input(user_question):
  emb=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key="AIzaSyDaFygGK9ocbwn1JRNKrB5_4H59dXmd8Dg")
  new_db=FAISS.load_local("faiss_index",emb,allow_dangerous_deserialization=True)
  docs=new_db.similarity_search(user_question)
  chain=get_conv()
  res=chain(
    {"input_documents":docs,"question":user_question},return_only_outputs=True
  )

  print(res)
  st.write("Reply:",res["output_text"])



def main():
  st.header("Chat with PDFs")
  user_question=st.text_input("Ask question here")
  if user_question:
    user_input(user_question)

  with st.sidebar:
    st.title("Menu :")
    pdfs=st.file_uploader("Upload your pdf files",type="pdf",accept_multiple_files=True)
    if st.button("Proceed"):
      txt=get_text(pdfs)
      chun=get_chunks(txt)
      get_vect(chun)
      st.success("Done")

if __name__ == "__main__":
  main()
