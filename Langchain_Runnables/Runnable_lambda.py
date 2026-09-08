#RunnableLambda is a primitive that allows to convert a custom python function as runnable and can use within an AI pipline like other runnables.
# let's say user created one function which includes some preprocessing data steps. now if we will use Runnable Lmabda that will make it a runnable and can directly connect with say LLm runnable and like wise.
# take tpoic-->first print joke an d second print total no. of words in joke

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

#function to count the nu. of words
def count_word(text):
    return len(text.split())

passthrough = RunnablePassthrough()

prompt = PromptTemplate(
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

joke_gen_chain = RunnableSequence(prompt,model,parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough,
    'word_count':RunnableLambda(count_word)   # can also write -->RunnableLambda(lambda x : len(x.split()))
}) 

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)