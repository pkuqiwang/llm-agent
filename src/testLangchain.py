import weaviate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import BaseDocumentTransformer, Document

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
        base_url='http://ollama:11434/v1', 
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

    prompt = prompt_template.invoke({"language": "Italian", "text": "what is the date today?"})
    response = model.invoke(prompt)
    print(response.content)

    prompt = prompt_template.invoke({"language": "Chinese", "text": "what is the day today?"})
    response = model.invoke(prompt)
    print(response.content)

def fileLoadSplitter() -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    
    return text_splitter.split_documents(PyPDFLoader("/data/llm.pdf").load())

   
#test file loading to Weaviate
def fileLoadingWeaviate():
    all_splits = fileLoadSplitter()

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://ollama:11434',
    )
        
    #store vector to Weaviate
    weaviate_client = weaviate.connect_to_local(host="weaviate", port=8080)

    db = WeaviateVectorStore.from_documents(
        all_splits, 
        embeddings, 
        client=weaviate_client)
    
    query = "How does self-supervised training work in LLM?"
    docs = db.similarity_search(query)

    # Print the first 100 characters of each result
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:100] + "...")

    weaviate_client.close()

#test file loading to Qdrant
def vectorLoadingQdrant():
    all_splits = fileLoadSplitter()
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://ollama:11434',
    )
        
    #store vector to Qdrant
    client = QdrantClient(url="http://qdrant:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        embedding=embeddings,
    )

    ids = vector_store.add_documents(documents=all_splits)
    client.close()

#test query loading to Qdrant
def vectorQueryQdrant():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url='http://ollama:11434',
    )
    client = QdrantClient(url="http://qdrant:6333")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        embedding=embeddings,
    )
 
    query = "How does self-supervised training work in LLM?"
    docs = vector_store.similarity_search(query)

    # Print the first 100 characters of each result
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(doc.page_content[:100] + "...")

    client.close()
