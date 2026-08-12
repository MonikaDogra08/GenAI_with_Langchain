from langchain_openai import OpenAI
from dotenv import load_dotenv 
# load the secret API key from .env file

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct') 
# object of openai with specific model

result = llm.invoke("what is the capital of India?") 
# hit the invoke method with our "prompt"

print(result)