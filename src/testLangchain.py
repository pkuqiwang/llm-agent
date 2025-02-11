import weaviate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_core.documents import BaseDocumentTransformer, Document
import os 

from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

#test with using OpenAI API with ollama models
def chatModel():
    model = ChatOpenAI(
        model="llama3.2:3b", 
        base_url='http://localhost:11434/v1', 
        api_key='ollama')

    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]
    returnMsg = model.invoke(messages)
    print(returnMsg.content)

#test with system prompt
def promptTemplate():
    system_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    model = ChatOpenAI(
        model="llama3.2:3b", 
        base_url='http://localhost:11434/v1', 
        api_key='ollama')
    
    prompt = prompt_template.invoke({"language": "Italian", "text": "what is the date today?"})
    response = model.invoke(prompt)
    print(response.content)

    prompt = prompt_template.invoke({"language": "Chinese", "text": "what is the day today?"})
    response = model.invoke(prompt)
    print(response.content)

#utility function
def fileLoadSplitter() -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.dirname(dir_path)
    pdf_path = parent_path + "/data/llm.pdf"
    return text_splitter.split_documents(PyPDFLoader(pdf_path).load())

#test vector loading to Weaviate, 
def vectorLoadingWeaviate():
    #this does NOT work, need update, use testWeaviate.testLoadWeaviate instead
    all_splits = fileLoadSplitter()
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://localhost:11434',
    )
    collection_name = "llmtest"
    text_key = "weaviatetestkey"
    weaviate_client = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051,)
    if weaviate_client.collections.exists(collection_name):
        weaviate_client.collections.delete(collection_name)

    vector_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name=collection_name,
        text_key=text_key, 
        embedding=embeddings, 
    )
    vector_store.add_documents(documents=all_splits)
    weaviate_client.close()

#test vector query to Weaviate
def vectorQueryWeaviate():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://localhost:11434',
    )
    collection_name = "llmtest"
    text_key = "weaviatetestkey"
    weaviate_client = weaviate.connect_to_local(host="127.0.0.1", port=8080, grpc_port=50051,)
    vector_store = WeaviateVectorStore(
        client=weaviate_client,
        index_name=collection_name,
        text_key=text_key, 
        embedding=embeddings, 
    )    
    
    query = "How does self-supervised training work in LLM?"
    docs = vector_store.similarity_search(query)

    # Print the first 100 characters of each result
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content)

    weaviate_client.close()

#test file loading to Qdrant
def vectorLoadingQdrant():
    all_splits = fileLoadSplitter()

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://localhost:11434',
    )
  
    #store vector to Qdrant
    qdrant_client = QdrantClient(url="http://127.0.0.1:6333")
    collection_name = "test_collection"
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
        
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    ids = vector_store.add_documents(documents=all_splits)
    qdrant_client.close()

#test query loading to Qdrant
def vectorQueryQdrant():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://localhost:11434',
    )

    qdrant_client = QdrantClient(url="http://127.0.0.1:6333")
    collection_name = "test_collection"
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
 
    query = "How does self-supervised training work in LLM?"
    docs = vector_store.similarity_search(query)

    # Print the first 100 characters of each result
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content)

    qdrant_client.close()
