# To ensure all components follow a standardized interface (i.e., the same set of rules and method calls), we use Runnables.
# Runnables are defined as an abstract base class (ABC).
# This abstract class declares a single method, invoke(), marked with @abstractmethod.
# The invoke() method itself does nothing in the abstract class — it simply enforces that any subclass must implement it.
# Now, all components commonly used in LLM application development — such as LLM chains, prompt templates, text splitters, PDF readers, and many others — inherit from this abstract class.
# Because they inherit from the same base, each component is required to define its own version of the invoke() method. This guarantees that every component exposes a common standardized method (invoke()) for producing the final output.
# The functionality of each component remains the same as before; the only difference is that they now share a unified interface. This makes the system more consistent, modular, and easier to integrate.

# so in short
# Abstract class (Runnable) → defines invoke() as a required method.
# Components inherit Runnable → must implement invoke().
# Result → all components are standardized with a common entry point.

# there are two types of runnables:
# 1.Task-Specific Runnables :These are individual components such as prompts, LLM models, parsers, and other modules.
# They perform their own specialized tasks (e.g., generating text, parsing outputs, formatting prompts).
# At the same time, they can also act as runnables within a pipeline, meaning they conform to the standardized interface (invoke()) and can be chained together seamlessly.

# 2.Runnable Primitives : These are utility constructs that help connect multiple task-specific runnables together.
# They define how components interact — for example:
# Sequentially: run one after another.
# In parallel: run simultaneously.
# Conditionally: run based on certain rules or conditions.(There are other many runnable permitive are available)
# Runnable primitives provide the structure and flow control for building complex pipelines.

import random
class Runnable(ABC):

  @abstractmethod
  def invoke(input_data):
    pass

# say if earlier LLM class was using predict() method for final response now after inheritance from abstract class(i.e.Runnable ) it will have invoke() in plac of predict() with same functionality.
# will be a task specific runnable
class LLM(Runnable):

  def __init__(self):
    print('LLM created')

  def invoke(self, prompt):
    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}


  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}

# same way for another component as well: now it has invoke() which was format() earlier
# will be a task specific runnable
class NakliPromptTemplate(Runnable):

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def invoke(self, input_dict):
    return self.template.format(**input_dict)

  def format(self, input_dict):
    return self.template.format(**input_dict)
  
# will be a task specific runnable
class NakliStrOutputParser(Runnable):

  def __init__(self):
    pass

  def invoke(self, input_data):
    return input_data['response']
  
#RunnableConnector which can help to connect any no. of component together like a chain by using runnable_list : output of 1 will act as input for other
# wiill be a Runnable permitives (like runnable sequence)
class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

            return input_data

template = NakliPromptTemplate(
template='Write a {length} poem about {topic}',
input_variables=['length', 'topic']
)

llm = NakliLLM()
parser = NakliStrOutputParser()
chain = RunnableConnector([template, llm, parser])
chain.invoke({'length':'long', 'topic':'india'})

# likewise can also connect two or no. of runnables together where output from one will go as input for other.