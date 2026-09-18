# A retriever is a component in Langchain taht fetches relevant documents from a data source(vectore DB) in response to a user's query
# so basically it behaves like a search engine which takes user's query in input and return no. of doc object relevent to that query
# internally it is going inside data source(db) checking for simmilarities and fetcheing those documnets
# All retrievers in Langchain are itself is Runnables

#1.data sorce retrievers : i) Wikipedia Retriever
from langchain_community.retrievers import WikipediaRetriever
# WikipediaRetriever hit the wikipedia API to get the relevent result for user's query
# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")  # we want top 2 relevent docs for user query, lang= will be en-->english

# Define your query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

print(docs)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display

#ii)Vector Store Retriever: fetch doc from vector store on sementic similarity using vector embeddings
# store your documents in a vector stores(FAISS, chroma,weaviate)  # chroma is open source and used mostly
# each doc is converted into dence vector using embedding model
# when user enters a query -->it also turned into a vector--->the retriever compare the query vector with the stored vectors-->and retrieves the top K most similar ones


from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Step 2: Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# AS we know that the same search thet retriever is doing can be done direclty using vectore stroes-->similarity_serach
# then why we creted this retriever?---> answer vector store can do similarity_search only based on sementic serach what if we want other strategy(advance) to do this similarity
# that options are available in Retrievers.
results = vectorstore.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


#type2 : Based on serach strategy--->i) MMR , ii) Multiqury iii) Contexual compression: 
#MMR (Max marginal relevance): return diverse perspective : return nor only relevent but aslo different from each other.
# let's use FAISS here instaed chroma.
# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

from langchain_community.vectorstores import FAISS


# Step 2: Create the FAISS vector store from documents
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)
#lambda_mult : plays an important role here if it is "1" then perform the similarity search like type 1 retrievers---> more towards 0 will give use more diverse results(different from each other)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

## Multi-Query Retriever: when user's query is not clear --> then it helps to generate multiple queries from main query like try to remove ambiguity from main query --> then return the results

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import MultiQueryRetriever

# Relevant health & wellness documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Create FAISS vector store
vectorstore = FAISS.from_documents(documents=all_docs, embedding=embedding_model)
# Create retrievers--> normal retriever will do the similarity search
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# multiquery_retriever
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")   # will use a LLM model here to generate multiple queries from main query
)

# Query
query = "How to improve energy levels and maintain balance?"
# Retrieve results
similarity_results = similarity_retriever.invoke(query)
multiquery_results= multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

print("*"*150)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


# Contextual Compression Retriever: Is an advanced retriever that improves retrieval quality by compressing documnet after retrieval:
# If say we have a documnet with mix concepts(mixed info) if will trim all the irrelavent things and keep text only related to the user query and return.
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain_core.documents import Document

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Create a FAISS vector store from the documents
vectorstore = FAISS.from_documents(docs, embedding_model)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # simple retriever
# Set up the compressor using an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")  #  LLM model to trim the irrelavant docs(takes query as well) got from simple retriever
compressor = LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


#***********There are no. of other advanced retriever to imporve the normal retrievers results if tehry are not that satisfactory (https://docs.langchain.com/oss/python/integrations/retrievers)*********************