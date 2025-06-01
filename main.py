import os
import streamlit as st
import pickle
import time
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
# st.set_option("logger.enableStaticWatchdog", False)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
# Load environment variables
load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize LLM
@st.cache_resource
def load_llm():
    llm = LlamaCpp(
        model_path="./models/llama-2-7b-chat.gguf",
        n_ctx=4098,
        max_tokens=500,
        temperature=0.7,
        verbose=True,
    )
    return llm

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = load_llm()

# if process_url_clicked:
#     # load data
#     #loader = UnstructuredURLLoader(urls=urls)
#     if urls:  # Make sure it's not empty
#         loader = UnstructuredURLLoader(urls=urls)
#     else:
#         st.error("No valid URLs provided.")

#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load(urls)
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

if process_url_clicked:
    if urls and isinstance(urls, list) and len(urls) > 0:
        try:
            main_placeholder.text("Data Loading...Started...✅✅✅")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
        except Exception as e:
            st.error(f"Data loading failed: {e}")
            st.stop()

        # Continue processing...
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
    else:
        st.error("No valid URLs provided.")


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            #result = chain({"question": query}, return_only_outputs=True)
            result = chain.invoke({"question": query})

            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
