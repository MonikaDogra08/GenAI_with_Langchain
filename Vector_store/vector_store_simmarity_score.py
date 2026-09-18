from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document #is a data structure (a class) used in LangChain to represent a piece of text along with optional metadata.


embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# after running this it will create my_chroma_db and have chroma.sqlite3
# add documents
vector_store.add_documents(docs)

# view documents
print(vector_store.get(include=['embeddings','documents', 'metadatas']))    # include used to check what we want to see in the output 

# search documents
vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2                                      # Get max 2 doc which are simmilar
)

# search with similarity score
vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)   # less is the score values more is the simmilarity(less distance)

# meta-data filtering : can also give facility to filter based on meta data
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

# update documents : Can update the existing doc with new text. We need to have document_id for that
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])

# delete document : Can also delete a doc
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# view documents
vector_store.get(include=['embeddings','documents', 'metadatas'])