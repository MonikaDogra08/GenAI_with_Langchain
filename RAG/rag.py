# Rag based system:
# Rag is a way to make a language model(like chatGPT) smarter by giving it extra information at the time you ask your question
# indexing includes following steps: (document ingestion(loader)-->text chunking--->Embedding generation--->store in a vector store): 
# indexing is a process of preparing external knowledge base --> which help to drive a context to pass into prompt
# Retrival :Based on query(change this query aslo into embeddings) --> check the context related to query and retrieve the most relevant piece of info out of it based on similarity scores
# Augmentation: query + retrieved doc(data)--> creates a prompt using both.(better to mention explicitly in the step the prompt this way to avoid hallicination:)

"""You are a helpful assistant.
Answer the question Only from the provided context. if the context is insufficient, just say you don't know.

{context}
Question: (question/query)"""

# Generation : LLm take this prompt---> genertes a response.

#***Rag based system is a simple and cheaper solution as compare to fine tunning your s/m where no training of model is required****

# Finally how we can evaluates these based solution to check thier accuracy :
# a)Can use Rages lib to evalutae the following matrics
#1) faithfulness : is the answer grounded in the retrieved context?
#2) answer_relevancy : is the answer relevant to user's question?
#3) context_precision : How much of the retrieved context is actually useful?
#4) context_recall : Did we reyrieve all neceesary information?

#b) Langsmith : Can trace the whole pipleline how and is it working as expected or not