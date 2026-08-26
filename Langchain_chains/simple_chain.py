from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)


model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template= 'Generate 5 interesting fact about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic':'cricket'})

result = result.replace("<|user|>", "You:")
result = result.replace("<|assistant|>", "AI:")
result = result.replace("</s>", "")
print(result)
# we can visualise chain with the help of following code:
chain.get_graph().print_ascii()