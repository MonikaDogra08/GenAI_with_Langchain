# Sometimes a document contains different topics in each paragraph. If we simply split by length, paragraph, or structure, we may lose the true semantic meaning because the chunks don’t align with the actual ideas in the text.
# How Semantic Splitting Works
# Instead of relying only on text structure (paragraphs, lines, or characters), semantic splitting looks at the meaning of sentences.
# It calculates the cosine similarity between consecutive sentences.
# When the similarity score drops significantly, it indicates a topic shift.
# At that point, the splitter creates a new chunk, ensuring each chunk contains text that is semantically coherent.
# still thery are in experimental stage not in so use...

from langchain_text_splitters import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
# from dotenv import load_dotenv

# load_dotenv()

text_splitter = SemanticChunker(
    embedding, breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)

