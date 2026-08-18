from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)

## Convert HuggingFaceEndpoint into a Chat model
model =ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content= "You are a helhful assistant")
] 

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content= user_input))
    if user_input== 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print("AI: ",result.content)

print(chat_history)