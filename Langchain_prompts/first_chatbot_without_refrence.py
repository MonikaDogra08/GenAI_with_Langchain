from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "do_sample": True,
#         "temperature": 0.3
#     }
# )
#******TinyLlama/TinyLlama-1.1B-Chat-v1.0 : model is not working correctly for this application so i am changing it with another chatmodel available in hugging face lib.

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)


model =ChatHuggingFace(llm = llm)

chat_history = []  #  keep appending the chat messages in this list
# if user says exit then break this conversation else go infinite
while True:
    user_input =  input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    chat_history.append(result.content)
    print("AI: ",result.content)

print(chat_history)

# There are few issues when we will run our code again then i won't able to recall the context of  previouse conversation done i.e the memory issue.
# second issue :every messages are getting appneded in this list what not in a defined manner like
# which message was send by user and which response sended my AI.
# so ideally it should mentain a dict like in a proper defined manner like:
# User:  user msg
# AI:  AI response
#.......(explaining in details in messages.py file)

# there are built in classes which takes care of this issue, we just have to import them and use.