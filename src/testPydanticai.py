from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

class CityLocation(BaseModel):
    city: str
    country: str

def testPydantic():
    ollama_model = OpenAIModel(
        model_name='llama3.2:3b',
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )
    agent = Agent(model=ollama_model, result_type=CityLocation)
    result = agent.run_sync('What is capital of United States?')
    print(result.data)
    print(result.usage())