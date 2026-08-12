from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model =  ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
# temp---> ranges from 0 to 1.5+ (value more towards 1.5+ will give more random/creative values)
# max_completion_tokens ---> no. of token we will get in output--->more no. of token---> more money to pay(for paid moels)
result = model.invoke("What is the capital of india?")

print(result)

# will not get only plain text as output but will get other meta data as well.

# to see our actual result we will print:
print(result.content)