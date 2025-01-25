import nest_asyncio
import pydantic_ai

def testPydantic():
    nest_asyncio.apply()

    from pydantic_ai import Agent, ModelRetry, RunContext, Tool
    from pydantic_ai.models.ollama import OllamaModel
    from colorama import Fore

    ollama_model = OllamaModel(
        model_name='llama3.2:3b',
        base_url='http://ollama:11434/v1',
    )
    agent = Agent(model=ollama_model)
    print(agent.run_sync('What is capital of United States?').data)
