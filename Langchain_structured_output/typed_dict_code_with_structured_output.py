# In this code we will be generating a summary and sentiment of the review summited using TypedDict using a LLm model

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.3,
    do_sample=True
)

## Convert HuggingFaceEndpoint into a Chat model
model = ChatHuggingFace(llm=llm)

#schema 1st way:
class Review(TypedDict):
    summary :str
    sentiment :str

#2nd way: as our llm our intelligent enough to understand is we mention to generate summary or sentiment out of the given text
# but if we want to make it more specific or defined from our side as well so we can use "Annotated"
# which can help model to get a clear instruction if anychance it has any doubt.

# class Review(TypedDict):
#     summary: Annotated[str, "A brief summary of the review"]
#     sentiment: Annotated[
#         Literal["positive", "negative", "neutral"],
#         "Sentiment of the review"
#     ]


structured_model = model.with_structured_output(Review)
# behind the scene this with_structured_output method generates a prompt in such way that
# it include the keys and ask to return a json forma in return from llm model and that llm model can generate the output as expected.
v 
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
""")

print(result)
print(result["summary"])
print(result["sentiment"])