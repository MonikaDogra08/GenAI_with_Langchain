# A tool is just a Python function(or API) that is packaged in a way the LLM can undersatnd and call when needed.
# There two types of tools: 1)Built-in tools 2)custom tools

#Built-in-tools : Already made tools in langchain.
#1)DuckDuckGoSearchRun: Can do realtime google/internet search
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke('top news in india today')

print(results)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

#2)shell tool: can exceute commond on commond line

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('whoami') # will show username

print(results)

# ******There are no. of tools available in langchain can check in documnetation.*******