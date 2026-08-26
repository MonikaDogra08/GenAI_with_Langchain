# One of drawbacks that json_output parser has it don't allow the schema enforce means can't get the json data based on predefined schemas that can be acheived using structured_output parsers
# we can enforce a schema in this methods based on that LLm can give us the results in that format.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema  
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
#StructuredOutputParser is not available in some newer LangChain versions so this code is not genearting any output

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)
# we have to create a desired schema with the help of ResponseSchema by defining proper name and other parameters

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


# prompt = template.invoke({'topic':'black hole'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)

# using chain 
chain = template | model | parser
result =  chain.invoke({'topic':'black hole'})
print(result)
