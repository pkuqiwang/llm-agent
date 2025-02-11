import weaviate
from weaviate.classes.config import Configure
from testLangchain import fileLoadSplitter

def testLoadWeaviate():
    all_splits = fileLoadSplitter()
    client = weaviate.connect_to_local(host="127.0.0.1",  
                                       port=8080,
                                       grpc_port=50051,
                                       )
    collection_name = "llmtest"
    text_key = "weaviatetestkey"
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    client.collections.create(
        collection_name,
        vectorizer_config=[
            Configure.NamedVectors.text2vec_ollama(
                name="nomic",
                source_properties=["ollam"],
                api_endpoint="http://ollama:11434",  #must use ollama as this is inside docker network, don't use localhost
                model="nomic-embed-text:latest",                 
            )
        ],
    )

    collection = client.collections.get(collection_name)
    data_rows = [{"title": f"Object {i+1}"} for i in range(5)]
    with collection.batch.dynamic() as batch:
        for data in all_splits:
            batch.add_object(
                properties = {text_key: data.page_content,},
            )

    for item in collection.iterator():
        print(item.uuid, item.properties)

    client.close()