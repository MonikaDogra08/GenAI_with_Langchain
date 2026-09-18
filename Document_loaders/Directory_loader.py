# Used to load multiple documents(tsxt files,pdf's) from a folder of files.
# it loads all docs at once in memory
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',    # mention the files wants to load (.pdf only)
    loader_cls=PyPDFLoader  # which doc_loader we wants to use to get the docs
)

docs = loader.load()
print(docs[325].page_content)
print(docs[325].metadata)

# while running this loader is taking time because it has 3 books with no. of pdfs and loading them directly on RAM.
# what if we have 1000 pdf -->then it will take a lot time to load
# Then lazy_load loads the doc in the menory but --> loads on demand---> it won't return list of docs---> but return a generator of doc
# it load one doc in memory as needed ---> do the operation then load the second doc--->so on

docs = loader.lazy_load()
for document in docs:
    print(document.metadata)