from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
# JsonOutputParser:forces an LLm to give it's output in json format.

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()
template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
# format_instruction : is a traditional insrtuction(which will be given by .get_format_instruction() method to tell what type of output we want from LLM

# one way to do:
# prompt = template.format()  
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))

# second way to do:
chain = template | model | parser
result = chain.invoke({}) # if no input variable is there to pass them pass empty {} else will give an error.
print(result)

