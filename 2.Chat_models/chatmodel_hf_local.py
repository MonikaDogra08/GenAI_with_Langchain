# we will be downloading the model locally nad running in this module

from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from huggingface_hub import constants

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature =0.5,
        max_new_tokens =100    )
)
print(constants.HF_HUB_CACHE)
#You can check the exact cache location with
 
model =ChatHuggingFace(llm = llm)

result = model.invoke("What is the capital of  india?")
print(result.content)