import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Step 1: Download required NLTK resource packages(only once)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

text = "The American Psycho movie was a classic thriller starring the protagonist Patrick Bateman."

# Step 2: Split the text into individual words (Tokenization)
words = word_tokenize(text)
print(f'Original sentence : {words}')

# Step 3: Assign a grammatical tag to each word (POS Tagging)
pos_tags = pos_tag(words)
print(f'POS TAGS :\n {pos_tags}')

# Step 4: Extract named entities using the pre-trained NLTK model
entity_tree = ne_chunk(pos_tags, binary=False)
print(f'Tree : {entity_tree}')

# Step 5: Parse the resulting tree to cleanly print out the entities
print("\n--- Identified Named Entities ---")
for subtree in entity_tree:
    if isinstance(subtree, nltk.Tree):
        entity_name = " ".join([token[0] for token in subtree.leaves()])
        entity_type = subtree.label()
        print(f"Entity: {entity_name} | Label: {entity_type}")

