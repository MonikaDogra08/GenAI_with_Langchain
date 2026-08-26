# In this example we will doing parallel chain operation where we will use two different LLM models
# with th ehlep of one we will generate simple notes from the text provided
# with the help of second model will generate the quiz.
# to implement parallel chain we need parallel runnables

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

load_dotenv()
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model_1 = ChatHuggingFace(llm = llm)

model_2 = ChatGoogleGenerativeAI(model= 'gemini-3.5-flash')

prompt_1 = PromptTemplate(
    template= 'Generate the short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template='Generate 5 short question answer from the following text \n {text}',
    input_variables=['text']
)

# final prompt where we will merge both outputs
prompt_3 = PromptTemplate(
    template= 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz ->{quiz}',
    input_variables=['notes','quiz']
    )

parser = StrOutputParser()

# define both chains
parallel_chain = RunnableParallel({
    'notes': prompt_1 | model_1 | parser,
    'quiz' : prompt_2 | model_2 | parser
})

# merge chain--> will be simply a sequancial chain --> where we can use any model out of 2
merge_chain = prompt_3 | model_1 | parser

# final_chain
chain = parallel_chain | merge_chain 

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""
result = chain.invoke({'text': text})

def clean_response(text):
    text = text.replace("<|user|>", "You:")
    text = text.replace("<|assistant|>", "AI:")
    text = text.replace("</s>", "")
    return text.strip()

result = clean_response(result)
print(result)

chain.get_graph().print_ascii()