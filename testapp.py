import warnings
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import ctransformers


API_KEY = "AIzaSyDxN5Y0MZFh0CyJ321YlvUK6SZFj4qdKRo"

warnings.filterwarnings("ignore")

# Models
#model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=API_KEY, temperature=0.5, convert_system_message_to_human=True)

# PDF Loader
def pdf_loader(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    return pages

# Splitter
def splitter(pages):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    return texts

# Embed
def embed(texts):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    return vector_index


llm=ctransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                  model_type='llama',
                  config={'max_new_tokens':512,
                          'temperature':0.8})



def qna(vector_index, question):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# Initialize the RetrievalQA chain with llama model and vectorstore
    qa_chain= RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})
    return result['result']



# Q&A Function

# Streamlit App
def main():
    st.title("Medical Chatbot")

    # Load the fixed PDF file
    pdf_path = r"Medical_book.pdf"
    
    # Load the PDF file
    pages = pdf_loader(pdf_path)
    texts = splitter(pages)
    vector_index = embed(texts)

    st.write("### You (Patient) Section:")
    question = st.text_area("Ask a question")

    if st.button("Send"):
        st.write("### Doctor's Reply Section:")
        st.write("Loading...")

        # Call your Q&A function here with the provided question
        answer = qna(vector_index, question)

        st.write(f"**Your Question:**\n{question}")
        st.write(f"**Doctor's Reply:**\n{answer}")

if __name__ == "__main__":
    main()
