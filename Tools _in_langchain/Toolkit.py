#Toolkit : Collection of related tools that serve a common purpose and give facility of reusability.


from langchain_core.tools import tool

# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

# create a class to get all the tools
class MathToolkit:
    def get_tools(self):
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools() # Can check all the tools available

for tool in tools:
    print(tool.name, "=>", tool.description)
