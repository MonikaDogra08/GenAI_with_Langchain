# A custom tool that you define yourself
# use them when :i.you want to call your own API's ii. Want to encapsulate business logic iii.want the LLm to intract with your DB, product or app
# there are no. of ways to create tools aminly used ones are these three:
# 1) using @tool decorator 2) using structuredTool & Pydantic 3) using BaseTool class

# 1) using @tool decorator :
from langchain_core.tools import tool
# Step 1 - create a function :adding a fucnction description is very important

def multiply(a, b):
    """Multiply two numbers"""
    return a*b

 #Step 2 - add type hints is also important

def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

# Step 3 - add tool decorator : is doing all the magic 

@tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b
# this tool is a runnable as well now
result = multiply.invoke({"a":3, "b":5})
print(result)
print(multiply.name) # function name 
print(multiply.description) # give description of this function
print(multiply.args)

#2) using structuredTool & Pydantic (follow a strict schema)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

#Pydantic schema:
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

# fucntion which will perform the action:
def multiply_func(a: int, b: int) -> int:
    return a * b

# main part:
multiply_tool = StructuredTool.from_function(
    func=multiply_func,   #  which fucntion to use
    name="multiply",     # name of function
    description="Multiply two numbers",# description
    args_schema=MultiplyInput   #schema
)

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)

#3)using BaseTool class : is an abstract class in for tools in langcahin.@tool structuresTool also inherit BaseTool class only.
from langchain.tools import BaseTool
from typing import Type

# arg schema using pydantic
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")

# Create a class which will inherit the BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:   # here function name can be  _run() only can't given any other name
        return a * b
    
multiply_tool = MultiplyTool()
result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)