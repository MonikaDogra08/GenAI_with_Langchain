from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Literal,Optional
from pydantic import BaseModel,Field
import json
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-32B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)
# ****************this model didin't support  :Pydantic schema is not supported for (with_structured_output)function calling *******************************

## Convert HuggingFaceEndpoint into a Chat model
model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] =  Field(description="Sentiment of the review")
    name : Optional[str] = Field(default=None, description="Write the name of the reviewer")

structured_model = model.with_structured_output(Review)
v 
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
""")

print(result)
print(result["summary"])
print(result["sentiment"])

# # Mention all the instruction in the prompt itsellf  
# prompt = """
# Analyze the following review.

# Return ONLY valid JSON.
# Do not include markdown.
# Do not include ```json.

# The JSON must have exactly these fields:

# {
#     "summary": "brief summary",
#     "sentiment": "positive",
#     "name": null
# }

# Review:

# I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say,
# it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes
# everything lightning fast—whether I’m gaming, multitasking, or editing
# photos. The 5000mAh battery easily lasts a full day even with heavy use,
# and the 45W fast charging is a lifesaver.

# The S-Pen integration is a great touch for note-taking and quick sketches,
# though I don't use it often. What really blew me away is the 200MP camera—
# the night mode is stunning, capturing crisp, vibrant images even in low
# light. Zooming up to 100x actually works well for distant objects, but
# anything beyond 30x loses quality.

# However, the weight and size make it a bit uncomfortable for one-handed
# use. Also, Samsung’s One UI still comes with bloatware—why do I need five
# different Samsung apps for things Google already provides? The $1,300
# price tag is also a hard pill to swallow.
# """


# # structured_model = model.with_structured_output(Review)
# response = model.invoke(prompt)

# print(response.content)
