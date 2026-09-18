# this is the simple and fast---> done based on length of text
# size we can define as characters(like 100 char at a time) or tokens
# https://chunkviz.up.railway.app/ ----> online application available for splitting based on different methods.
#limitation:CAn easily loose semnetic meaning of any words in between.
# chunk_overlap ---> can help to retain the semnetic meaning with next split (10 to 20% values we can keep to chunk overlap)


#document loader + text splitter

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,  
    separator=''
)

result = splitter.split_documents(docs)

print(result[1].page_content)