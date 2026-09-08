# will create a parallel chain where common input will go into saem LLM/different LLm -->one will generates a tweet and second will generates a linkdin post
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv

prompt_1 = PromptTemplate(
    template= "Generate a tweet about {topic}",
    input_variables= ['topic']
)

prompt_2 = PromptTemplate(
    template= 'Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

# model = ChatGoogleGenerativeAI(model= 'gemini-3.5-flash')
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm) 

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt_1,model,parser),
    'linkedin': RunnableSequence(prompt_2,model,parser)
})

result = parallel_chain.invoke({'topic': 'AI'})
print(result['tweet'])
print(result['linkedin'])