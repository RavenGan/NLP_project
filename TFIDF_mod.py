from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_qa(passage, question, sents_starts):
    # Split the passage into sentences
    sentences = separate_list(passage, sents_starts)
    # Include the question
    sentences.append(question)

    # Convert sentences into TF-IDF representation
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Get the vector for the question
    question_vector = vectors[-1].reshape(1, -1)

    # Compute cosine similarity between the question and every sentence
    cosine_similarities = cosine_similarity(question_vector, vectors[:-1]).flatten()

    # Get the sentence with the highest similarity as the answer
    answer_index = cosine_similarities.argmax()
    return answer_index

def separate_list(passage, sents_starts):
    # Separate a passage into sentences based on the starting indices.
    characters = list(passage) # Split the passage into characters
    sublists = []
    for i in range(len(sents_starts)):
        # If it's the last index, slice until the end of the main list
        if i == len(sents_starts) - 1:
            sub_char = characters[sents_starts[i]:]
            sentence = ''.join(sub_char)
            sublists.append(sentence)
        else:
            # Slice from the current index to the next index
            sub_char = characters[sents_starts[i]:sents_starts[i+1]]
            sentence = ''.join(sub_char)
            sublists.append(sentence)
    return sublists

def get_true_sent_idx(selected_sent, sents_start):
    # Get the correct selected sentence index
    start_idx = selected_sent['start']
    true_sent_idx = sents_start.index(start_idx)
    return true_sent_idx

def criterion(true_idx, pred_idx):
    if true_idx == pred_idx:
        return 1
    else: 
        return 0