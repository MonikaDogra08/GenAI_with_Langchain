from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)


model =ChatHuggingFace(llm = llm)

# Input to pass LLm model
messages = [
    SystemMessage(content = "You are a helhful assistant"),
    HumanMessage(content = "tell me about Langchain")
]

result = model.invoke(messages)
# append the final result as AI response in the list as well
messages.append(AIMessage(content = result.content))

print(messages)

# Now we will implement the same changes in out first_chatbot.py file to see the proper messages.
# ia will be making a second file to implement these changes first_chatbot_hf.py