import weaviate
from weaviate.classes.config import Configure

def testLoadWeaviate():
    client = weaviate.connect_to_local(host="weaviate", port=8080)

    if client.collections.exists("test_collection"):
        client.collections.delete("test_collection")

    client.collections.create(
        "test_collection",
        vectorizer_config=[
            Configure.NamedVectors.text2vec_openai(
                name="title",
                source_properties=["title"],
                base_url="http://ollama:11434", #don't include /v1 in the base url
                model="nomic-embed-text:latest", 
            )
        ],
    )

    collection = client.collections.get("test_collection")

    data_rows = [{"title": f"Object {i+1}"} for i in range(5)]
    with collection.batch.dynamic() as batch:
        for data_row in data_rows:
            batch.add_object(
                properties=data_row,
            )

    for item in collection.iterator():
        print(item.uuid, item.properties)

    client.close()