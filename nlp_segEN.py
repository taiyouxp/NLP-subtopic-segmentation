import spacy

# Loading the English language model
nlp = spacy.load("en_core_web_lg")

# 1. Preprocessing with spaCy
def pre_processing(text):
    doc_spacy = nlp(text)  # Processing the text by creating a 'doc' object containing tokens and linguistic information (POS, tagging, lemma)

    processed_sentences = []  # List to store processed sentences
    original_sentences = []  # List to store original sentences

    for sentence in doc_spacy.sents:  # .sents iterates over each sentence in the document
        original_sentences.append(sentence.text)  # .text stores the original sentence
        words = [
            token.text  # Tokenizes and keeps original text (not lemmatized)
            for token in sentence  # Iterates over each token in the sentence
            if not token.is_stop  # Removes stopwords
            and token.is_alpha  # Keeps only words (removes numbers and punctuation)
            and token.pos_ in ["NOUN", "VERB", "PROPN"]  # Uses .pos_ to filter nouns, verbs, and proper nouns
        ]
        if not words:
            words = ["_PLACEHOLDER_"]  # Add a placeholder for empty sentences
        processed_sentences.append(words)  # Adds to the processed sentences list

    print("Total sentences:", len(processed_sentences))
    print(processed_sentences, original_sentences)

    return processed_sentences, original_sentences

# 2. Sentence similarity function
def similarity(sentence_list):
    docs = [nlp(sent) for sent in sentence_list]  # Processes each sentence with spaCy (used to calculate similarity)
    similarities = []
    for i in range(len(docs) - 1):  # Iterates over doc indices except last, comparing each sentence with the next
        sim = docs[i].similarity(docs[i + 1])  # Calculates cosine similarity between consecutive sentences using embeddings
        similarities.append(sim)
        print(f"Similarity between sentence {i + 1} and {i + 2}: {sim:.2f}")

    return similarities

# 3. Keyword extraction
def extract_keywords(processed_sentence):
    keywords = []
    doc = nlp(" ".join(processed_sentence))
    for token in doc:
        if token.pos_ == "PROPN":
            keywords.append(token.text)
    # If no proper nouns, add other relevant words (we already have nouns, verbs, proper nouns)
    if not keywords:
        keywords = processed_sentence
    
    return list(set(keywords))[:5]  # Removes duplicates and limits to 5 keywords

# 4. Label generation
def generate_subtopic_label(processed_subtopic):
    # Here, processed_subtopic is a list of processed sentences (word lists)
    keywords = []
    for processed_sentence in processed_subtopic:  # Iterates over each processed sentence in subtopic
        keywords.extend(extract_keywords(processed_sentence))  # Adds extracted keywords to keywords list
        # extend() adds all elements of one list to another

    return list(set(keywords))[:5]  # Removes duplicates with set(), limits to 5 keywords with slicing '[:5]'
    # and converts set back to list with list()

# 5. Document segmentation
def document_segmentation(doc, similarity_threshold):
    processed_sentences, original_sentences = pre_processing(doc)  # Returns lists of processed and original sentences
    similarities = similarity(original_sentences)  # Calculates similarity between sentences

    groups = []  # List to store sentence groups
    current_group = [original_sentences[0]]  # Starts first group with first original sentence
    current_processed_group = [processed_sentences[0]]  # Starts first group with first processed sentence
    
    min_len = min(len(original_sentences), len(processed_sentences))  # Finds minimum length between the two lists
    for i in range(1, min_len):  # Iterates over remaining sentences
        if similarities[i - 1] >= similarity_threshold:  # If similarity is high
            current_group.append(original_sentences[i])  # Adds original sentence to current group
            current_processed_group.append(processed_sentences[i])  # Adds processed sentence to current group
        else:
            groups.append((current_group, current_processed_group))  # Finishes current group and adds to groups list
            if i < min_len -1:
                current_group = [original_sentences[i]]  # Starts new group with original sentence
                current_processed_group = [processed_sentences[i]]  # Starts new group with processed sentence

    groups.append((current_group, current_processed_group))  # Adds last group to list

    return groups

# 6. Main execution
with open('virgil.txt', 'r') as f:  # Opens text file
    text = f.read().replace('\n', ' ')  # Reads file content

threshold = 0.9  # Defines similarity threshold 
groups = document_segmentation(text, threshold)  # Segments document into subtopics

for i, (original_group, processed_group) in enumerate(groups):  # Iterates over sentence groups
    label = generate_subtopic_label(processed_group)  # Generates label for subtopic based on processed sentences
    print(f"\nSubtopic {i + 1} - Label: {', '.join(label)}")  # Displays label
    for sentence in original_group:  # Iterates over original sentences in subtopic
        print(f"{sentence}")  # Displays each sentence