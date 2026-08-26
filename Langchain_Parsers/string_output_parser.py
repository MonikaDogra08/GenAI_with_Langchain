# it simply return a string in output to the LLm's response
# when we run simple LLm model it gives use so many meta data along with actual response in result
# we use to fetch actual reponse using .content() at the end to avoid other meta data
# but with the help os string_output parser it will return only the content part no need to specifically use the .content() 
# we will do this implementation with TinyLlama/TinyLlama-1.1B-Chat-v1.0 

# this project is ---> take the topic from the user--->give to LLM to generate a detailed report on that topic----> Then again pass that detailed report to the same LLm  to give 5 line summary


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
# StrOutputParser --> it's common usage is with chains---> convert it into a pipeline

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)


model = ChatHuggingFace(llm = llm)

#1st prompt(detailed report)
template_1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)

#2nd prompt(summary)
template_2= PromptTemplate(
    template= "Write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

# simple flow without parser
# prompt1 = template_1.invoke({'topic':'black hole'})
# result = model.invoke(prompt1)
# prompt2 = template_2.invoke({'text':result.content})
# result1 = model.invoke(prompt2)
# print(result1.content)

# flow with parser:
parser = StrOutputParser()

# chain
chain = template_1 | model | parser | template_2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)