# Runnbale passthrough is a special runnable primitive that simply return the i/p as o/p without modifying it:
# say we are creating a seq. chain where we want to generate the joke as well as explaination of the topic---> it will do both things but finally will show only explaination in o/p cann't print joke
# what if we want to print/see the joke as well? ---> we can use the Runnable passthrough
# now we will create a seq chain which will generat a joke then create a parallel chain using passthrough one chain will return joke as it is (given as i/p)
# second chanin(where same joke passed as i/p) now will generate its explaination.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv

passthrough = RunnablePassthrough()

prompt_1 = PromptTemplate(
    template= ' Write a joke about {topic}',
    input_variables= ['topic']
)
# model = ChatGoogleGenerativeAI(model= 'gemini-3.5-flash')
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm) 

parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template='Explain the following joke- {text}',
    input_variables= ['text']
)

joke_genration_chain = RunnableSequence(prompt_1,model,parser)
parallel_chain =  RunnableParallel({
    'joke' : passthrough,
    'explaination' : RunnableSequence(prompt_2,model,parser)
})

final_chain = RunnableSequence(joke_genration_chain,parallel_chain)
result = final_chain.invoke({'topic','Cricket'})

print(result["joke"])
print(result["explaination"])