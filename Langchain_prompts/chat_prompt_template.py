# in this we will using multi dynamic prompts i.e. where system as well as human mgs can we dynamic:
# so now we have to use ChatPromptTemplate
# as we alreday used: PromptTemplate class which was used for single dynamic msg.

from langchain_core.prompts import ChatPromptTemplate

#ChatPromptTemplate : use for multi turn conversation place holders.
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')
 
])

prompt = chat_template.invoke({'domain':'cricket','topic': 'Free hit'})

print(prompt)

# as when we do a chat with chatbot: and we come back after some time: we need all the previous chat history to get the reference.
# so we need to store it in any DB and when a new chat will start we will load that first from the maintained chat history
# this problem is solved by a placeholder : which is called a message placeholder.
# for mow i will be storing the prevoius history of chat in some.txt file.