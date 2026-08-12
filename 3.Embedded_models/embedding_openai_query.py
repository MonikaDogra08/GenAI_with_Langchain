# generates the embedding vectors for single query

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding =  OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)
# "dim" here will generate the vector(32) of our query by capturing its sementic meaning
result = embedding.embed_query("Delhi is capital of india")

print(str(result))