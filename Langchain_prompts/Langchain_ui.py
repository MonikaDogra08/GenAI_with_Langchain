# create a Research tool with the help of streamlit Ui interface which will we able to summarize any paper given by user.
# will provide dynamic prompt to user to get the output
# will ask three things from :
#1) which paper user wants to summarize
#2)what type explainantion he wants
#3)how much explaination length user is expecting


import streamlit as st
from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from huggingface_hub import constants
from langchain_core.prompts import PromptTemplate,load_prompt

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
            temperature =0.5,
            max_new_tokens =300,
            do_sample = True ),
            
)

model =ChatHuggingFace(llm = llm)
st.header("Research tool")

# user_input = st.text_input("Enter your prompt") 
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# template
template  = load_prompt('template.json')

# fill the placeholders
# prompt = template.invoke({
#     "paper_input":paper_input,
#     "style_input":style_input,
#     "length_input":length_input
# })

# if st.button("Summarize"):
#     result = model.invoke(prompt)
#     st.write(result.content)

# instead of writing above commented code we can create a chain can avoid above code like:

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input
    })
    st.write(result.content)
# here we are calling invoke only once instead of calling it no. of times(used chain concept)

