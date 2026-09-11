import nltk
from nltk.tokenize import word_tokenize

# Download required resources (Run once!)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Input text
text = "If two pieces of the same type of metal touch in space, they bond permanently."
print(f"Original Text: {text}")

# Tokenization
tokens = word_tokenize(text)

# POS Tagging
pos_tags = nltk.pos_tag(tokens)
print(f"\nPOS Tags:\n{pos_tags}")

# Grammar for Noun Phrase
grammar = r"""
NP: {<DT>?<CD>?<JJ>*<NN.*>+} # Separates DT and CD properly
VP: {<VB.*>+<RB.*>*} # Matches verbs followed by optional adverbs
"""
# Create chunk parser
chunk_parser = nltk.RegexpParser(grammar)

# Perform chunking
chunk_tree = chunk_parser.parse(pos_tags)
print(f"\nChunked Output:\n{chunk_tree}")

# Display Noun Phrases
print("\n--- Extracted Phrases ---")
for subtree in chunk_tree.subtrees():
    # Change the condition to include 'VP'
    if subtree.label() in ['NP', 'VP']: 
        words = " ".join([word for word, tag in subtree.leaves()])
        print(f"[{subtree.label()}]: {words}")