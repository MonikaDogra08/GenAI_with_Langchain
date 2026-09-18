# PyPDf loader used when we have mostly textual data in the pdf ---> it is not great with scanned pdf and complex layouts
# there are no. of other pdf loader which are available.
# PyPDf is Mostly used documnet loader:it works page wise page basis
# if we have 25 pages in a pdf ---> then it create 25 document objects

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)