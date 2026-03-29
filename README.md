#MULTICLOUD AI POWERED PDF INTELLIGENCE
An AI-driven pdf analysis tool that uses retrieval augumented generation(RAG)to provide instant answers from uploaded documents.
###features
-**Multi-pdf support:**upload and process multiple documents simultaneously
**fast interference:**powered by llama 3.3 70B via Groq for near instant responses.
-**persistant vector database:**uses chroma db to store and retrieve document context.
_**clean UI:**built with streamlit,including a 'clear chat'function for privacy
###tech stack
-**language:**python
**orchestration:**langchain
-**LLM:**Llama3.3 70B(groq)
-**vector database:**chromaDB
-**Embeddings:**HuggingFace

##How to run
1.clone this repository
2.install requirements:'pip install -r requirements.txt'
3.Add your 'GROQ_API_KEY' to a '.env' file
4.run: 'streamlit run web_app.py