# In this application we will use the concept of conditional_chains :
# where based on useer's feedback we will clasify the sentiment of it.
# if response is +ve them generate or email/reply back to that custer as "Thank you msg" else as "inconvenience msg"
# point to note here only one step will get executed here as it is conditional chain not parallel responces.

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
# RunnableBranch --> with the help of it we can execute chains using if, else

load_dotenv()
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm = llm)   ---> this model doesn't support the defined format so used gemini-3.5-flash'
model = ChatGoogleGenerativeAI(model= 'gemini-3.5-flash')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['Postive', 'Negative'] = Field(description= 'Give the sentiment of the Feedback'
    )

## to keep the output from the model consistent --> like want only 'postive' not like "POS" or somthings else we can use pydantice parser
parser_2 = PydanticOutputParser(pydantic_object= Feedback)

prompt_1 = PromptTemplate(
    template='Classify the sentiment of the following text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables= {'format_instruction':parser_2.get_format_instructions()}
)

# this is our classification part
classifier_chain = prompt_1 | model | parser_2
print(classifier_chain.invoke({'feedback':' This is a wonderful smartphone'}).sentiment)

prompt_2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables= ['feedback']
)
prompt_3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables= ['feedback']
)
# conditonal branching--->it takes----> if else condiitons, chain
branch_chain = RunnableBranch(
    (lambda x :x.sentiment =='Postive', prompt_2 | model | parser),
    (lambda x :x.sentiment =='Negative', prompt_3 | model | parser),
    RunnableLambda(lambda x : 'Could not find sentiment')  # default condition
)

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback':' This is a wonderful smartphone'})
print(result)

chain.get_graph().print_ascii()