from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv

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

chain  = RunnableSequence(prompt_1,model,parser,prompt_2,model,parser)
print(chain.invoke({'topic':'AI'}))
