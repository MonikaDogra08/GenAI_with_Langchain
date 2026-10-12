# Tool binding: is the step where you register tools with a LLM so that:
#1. The LLM knows thta tools are available
#2. it knows what each tool does (via description)
#3. it knows what input format to use(via schema)

#*****Not all LLm has the capacity of tool binding*****
# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature =0.5,
        max_new_tokens =100    )
)

llm =ChatHuggingFace(llm = llm)   # this model is not able to make tool calls

# tool create

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))
print(multiply.name)
print(multiply.description)
print(multiply.args)

# tool binding
llm_with_tools = llm.bind_tools([multiply])

# Tool calling: LLM doesn't run the tool, it just suggests the tool and input argumnet.The actual execution is handled by Langchain or you

#print(llm_with_tools.invoke('can you multiply 3 with 10').tool_calls)

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]

# tool execution:
result = llm_with_tools.invoke(messages)
messages.append(result)
print("before:",messages)

tool_result = multiply.invoke(result.tool_calls[0])
print(tool_result)
messages.append(tool_result)
print("after:",messages)  # here we will have HumanMessage, AIMessage and ToolMessage
