# in this project we will will using no. of doc with no. of text and then user will send his query
# then we will check the query is related to which doc using cosine similarities

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)

document = [
    "Virat Kohli, born on 5 November 1988, is one of the most celebrated cricketers in the world",
    "MS Dhoni has an aggressive batting style, sharp cricketing mind, and unmatched consistency, he has broken numerous records across all formats of the game.",
    "Rohit Sharma is a former captain of the Indian national team, Kohli’s journey from a young boy in Delhi to a global sports icon is a story of dedication, discipline, and passion."
    ]

query = "Tell me about virat kohli"

# doc embeddings:
doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query(query)

# similarity: both list should be 2d
scores = cosine_similarity([query_embeddings],doc_embeddings)[0]
# will return similarity score of our query with every sentense we have in the document

# get the highest similarity score along with index no.
index,score  = sorted(list(enumerate(scores)),key = lambda x : x[1][-1])
# sort the similarity score based on score not index i.e x[1] and then[-1] to get the highest score 

print(query)
print(document[index])
print("similarity score is :", score)


#**** Now next step would be to store these similarity score so that we don't need to calculate these vectors again and again for siliar search that where the concept of vector store comes into picture.