# Process:
# fetch any youtube video's transcript---> can be done using various ways:
#  --> Can use langchai's YT loader 
#  ---> Using youtube's API--> hit Api and get the transcript(in this we use this)
# split text using textspitter---> then generate embeddings-->store in vector store--->create/use retriever-->create a prompt(augmentation)-->response

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

video_id = "Gfr50f6ZBvo"

try:
    # Create an instance of the API
    api = YouTubeTranscriptApi()
    fetched_transcript = api.fetch(video_id, languages=["en"])

    # Convert to list of dicts (same style as get_transcript)
    transcript_list = fetched_transcript.to_raw_data()

    # Print first few entries to check
    # print(transcript_list[:5])

    # Flatten to plain text if needed
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print("\nFull Transcript:\n", transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

# Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(len(chunks))
# print(chunks[0])

#Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings =  HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id[0])
# print(vector_store.get_by_ids(['4aae52cd-5d73-4f39-bf2a-f3c9efa09735']))

#Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # return top 4 realted output
# print(retriever)
# print(retriever.invoke('What is deepmind'))

# Step 3 - Augmentation
# create an LLM:

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature =0.5,
        max_new_tokens =100    )
)

llm =ChatHuggingFace(llm = model)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question  = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs  = retriever.invoke(question)
# print(retrieved_docs)
# instead of getting everything concate page_content part:
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)
# at this step we ahve our question(query) + content
final_prompt = prompt.invoke({"context": context_text, "question": question})
# print(final_prompt)

# Step 4 - Generation
answer = llm.invoke(final_prompt)
print(answer.content)


#********************Now there is only one problem we are calling all steps: indexing -->Retriever--->Augmenation -->response manually***********
# To avoid this we will build a chain:

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# 1st part:
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),    # question going into retriever --->retriever will do all the sementic search and fetch the relavent doc for asked question---> and process them into a combined contextdoc
    'question': RunnablePassthrough()                     # in paralll it need query as well 
})

# print(parallel_chain.invoke('who is Demis') ) # will give --> context and question

# 2nd part: output from parallel chain ie. context and question --> go into prompt ---> based on given context LLm will genearte a reponse
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
print(main_chain.invoke('Can you summarize the video'))