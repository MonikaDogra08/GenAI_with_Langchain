# A text structure–based splitter is one of the most commonly used methods for breaking text into smaller chunks.And better than Text spilter
# It works by recognizing the natural structure of the text and splitting accordingly.
# How It Splits
# Paragraphs (\n\n) → Splits whenever there is a blank line, treating each paragraph as a chunk.
# Lines (\n) → Splits at every line break, useful for structured text like logs or code.
# Words → Splits text into smaller word groups, often used when fine‑grained control is needed.
# Characters → Splits text into fixed‑size character chunks, ensuring consistent length regardless of structure.


from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)

