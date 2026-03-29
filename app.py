import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import RetrievalQA

#___setup___
os.environ["GOOGLE_API_KEY"]="AIzaSyAyfZbx783TWPcyencjYjevF9h1gwD-JbM"
print("Mentor Note:Reading your test.pdf...")

#1. load the pdfs
loader=PyPDFLoader("test.pdf")
data=loader.load()

#2.split text into chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=100)
docs=text_splitter.split_documents(data)

#3.create embeddings and vector store
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore=Chroma.from_documents(documents=docs,embedding=embeddings)

#4 create ai chain
llm=ChatGroq(model="llama-3.3-70b-versatile",api key="groq_api_key")
qa_chain=RetrievalQA.from_chain_type(llm,chain_type="stuff",retriever=vectorstore.as_retriever())

#5.phase 2-user interaction & final chat loop
print("\n---phase 2: chat mode enabled(type 'exit' to stop) ---")
while True:
     user_query=input("\nwhat would you like to know ?(or type 'exit'):")
#check if the user wants to quit
     if user_query.lower()=='exit':
        print("\nExiting chat.goodbye!")
        break
             
     print("AI is searching...")
     # get the answer
     result=qa_chain.invoke({"query":user_query})
     answer=result["result"]

     print(f"\n__AI ANSWER__\n{answer}")

#6 auto save to notepad . we use "a" for append so it will build a list for all your q&a
with open("qa_results.txt", "a", encoding="utf-8") as f:
    f.write(f"question:{user_query}\n")
    f.write(f"Answer: {answer}\n")
    f.write(f"{'-'*30}\n")
print("---success! this answer is saved in 'qa_results.txt' ---")