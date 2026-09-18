from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')
# used to load the CSV files ---> creates a doc object fro every row
docs = loader.load()

print(len(docs))
print(docs[1])