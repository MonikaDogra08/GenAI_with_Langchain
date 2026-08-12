# will use sntense tranformers model: it maps sentences/paragraphs to 384 dim dense vector space 
# can be used for clustering and sementic search etc.
# will download this model locally as well.

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of india"

# vector = embedding.embed_query(text)
# will genearte vectors for single text
query = "Tell me about delhi?"

documents = [
    "Delhi is capital of india",
    "Kolkata is capital of WB",
    "Paris is the capital of France"
    ]
vector = embedding.embed_documents(documents)
# Will generate vectors for docs

# Let's make and try the changes like we did in file document_similarity.py to get the score with our locally stored model.
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding],vector)[0]
print(scores)
# get the highest similarity score along with index no.
# print(sorted(list(enumerate(scores)),key = lambda x : x[1]))
index, score = max(enumerate(scores),key=lambda x: x[1])

print(query)
print(documents[index])
print("similarity score is :", score)
# print(str(vector))