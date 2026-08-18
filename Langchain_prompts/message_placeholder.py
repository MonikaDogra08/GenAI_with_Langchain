# Message PlaceHolder: we create a palceholder for set of messages to store or retrive the chat history.

from langchain_core.prompts import ChatPromptTemplate, MessagePlaceholder

#chat template:
chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer agent'),
    MessagePlaceholder(variable_name = 'chat_history'),
    ('human','{query}')
])

chat_history = []

# load the history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund?'})
# for chat_history placeholder we want to give the chat_history list that we store weth prevouse chats
print(prompt)