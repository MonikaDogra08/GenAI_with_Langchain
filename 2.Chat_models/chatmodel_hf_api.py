### Code to check wether the code is able to read or pick API key correctly from .env file or not and also to check the explicitly request the fastest provider

# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# load_dotenv()

# print("HF token loaded:", os.getenv("HF_TOKEN") is not None)

# client = InferenceClient(
#     api_key=os.getenv("HF_TOKEN")
# )

# response = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of India?"
#         }
#     ]
# )

# print(response.choices[0].message.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# HuggingFaceEndpoint----> used when we are using huggingface API interface

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "openai/gpt-oss-120b",
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # this model was causing issue while running from API so i changed the model
    task = "text-generation"
)
#repo_id repersents model name from hugging face
# here we are using "TinyLlama/TinyLlama-1.1B-Chat-v1.0" which is a kind of small model as compare to other so to run locally just prefering it.
# task---> parameter use to define the task we want to perform using this defined model.
model = ChatHuggingFace(llm= llm)

result = model.invoke("What is the capital of india?")
print(result.content)