from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from ollama import Client
from qdrant_client.http.models import PointStruct


def testLoadQdrant():
  ollamaClient = Client(host='http://ollama:11434')

  qdrantClient = QdrantClient(url="http://qdrant:6333")
  collection_name = "test_collection"
  if not qdrantClient.collection_exists(collection_name):
    qdrantClient.create_collection(
      collection_name=collection_name,
      vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
      )
  collection_info = qdrantClient.get_collection(collection_name = collection_name)

  documents = [
    "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
    "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
    "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
    "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
    "Llamas are vegetarians and have very efficient digestive systems",
    "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
  ]

  # store each document in a vector embedding database
  for i, d in enumerate(documents):
    response = ollamaClient.embed(model='nomic-embed-text:latest', input=d)
    qdrantClient.upsert(
      collection_name = collection_name, 
      wait=True,
      points=[
        PointStruct(id=i, vector=response.embeddings[0], payload={"text": [d]})
      ]
    )