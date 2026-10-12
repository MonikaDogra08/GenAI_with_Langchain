# Agent = LLM + erternal Tools

# from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub   # is a hub for number of pre written prompt, is basically like git hub for prompt

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature =0.5,
        max_new_tokens =100    )
)
 
model =ChatHuggingFace(llm = llm)

# Pull the ReAct prompt(decide which design pattern we want to use) from LangChain Hub(this hub has prewritten prompt for use)
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
# agent use to search for the available tools
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)
# Step 4: Wrap it with AgentExecutor (execute the instruction coming from agent)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool,get_weather_data],
    verbose=True
)

# Step 5: Invoke
response = agent_executor.invoke({"input": "3 ways to reach goa from delhi and aslo tell me goa's current weather condiiton"})
print(response)

print(response["output"])