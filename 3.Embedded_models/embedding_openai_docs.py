# # generates the embedding vectors of multiple queries or docs

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding =  OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)
# "dim" here will generate the vector(32) of our query by capturing its sementic meaning

documents = [
    "Delhi is capital of india",
    "Kolkata is capital of WB",
    "Paris is the capital of France"
    ]
result = embedding.embed_documents(documents)

print(str(result))