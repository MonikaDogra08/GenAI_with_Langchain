# Used for conditional chains
# topic--->prompt(ask to generate a summary about the topic)--->LLm--->parser--> one condiiton if summary is>500 words -->ask llm to short that summary with <500
# else is summary is already<500 words--->then print as it is.
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm = llm) 

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser =  StrOutputParser()

report_gen_chain = RunnableSequence(prompt1,model,parser)   # can use : prompt1|model|parser

# syntax for RunnableBranch is:
# RunnableBranch(
#     (condition,chain),  # here condition will be the out of report_gen_chain will go as input to this
#     (condition,chain),
#     default
# )
brach_chain = RunnableBranch(
        (lambda x:len(x.split())>300, RunnableSequence(prompt2,model,parser)), # can also use : prompt2|model|parser
        RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,brach_chain)
print(final_chain.invoke({'topic':'Russia Vs Ukrain'}))


#***LCEL (langchain expression langauge)
# As observed that seq.chain is somthing which is getting used every where so to make it easy:
# rather than using RunnableSequence(r1,r2,r3)--> we can replace it with pipeline operators like---> [r1 | r2 | r3]
