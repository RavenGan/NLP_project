import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Model(torch.nn.Module):
    def __init__(self, hidden_size=1024, device='cpu'):
        super(Model, self).__init__()
        # Load SpanBERT Tokenizer and Model
        self.tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
        self.model = AutoModel.from_pretrained('SpanBERT/spanbert-large-cased').to(device)

        self.hidden_size = hidden_size
        # Use attention mechanism concatenate the question and text hidden states
        self.question_att = nn.Linear(hidden_size, hidden_size)
        self.context_att = nn.Linear(hidden_size, hidden_size)
        self.qa_outputs = nn.Linear(hidden_size, 2)
    
    def cal_answer_f1(self, question_embeddings, text_embeddings, answer_start, answer_end):
        question_att = self.question_att(question_embeddings)
        context_att = self.context_att(text_embeddings)

        # Compute attention weights
        attn_weights = torch.matmul(context_att, question_att.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention weights
        attended_question = torch.matmul(attn_weights, question_embeddings)

        # Combine question and attended context
        combined_hidden = text_embeddings + attended_question

        # Predicting start and end logits
        logits = self.qa_outputs(combined_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        start_index = torch.argmax(start_logits).item()
        end_index = torch.argmax(end_logits).item() + 1

        # Use F1 score to measure the correct answer span
        if end_index <= start_index:
            f1_score = 0
        elif answer_end <= answer_start:
            f1_score = 0
        else:
            pred_span = set(range(start_index, end_index))
            true_span = set(range(answer_start, answer_end))
            overlap = len(pred_span.intersection(true_span))
            precision = overlap / len(pred_span)
            recall = overlap / len(true_span)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        return f1_score, (start_index, end_index)
    
    def sents_cos_similarity(self, question_embeddings, sents_embedding_ls, true_sent_idx):
        question_embeddings = question_embeddings[0, 1:-1]
        similarity_scores = []
        for sents_embedding in sents_embedding_ls:
            sents_embedding = sents_embedding[0, 1:-1]
            question_expanded = question_embeddings.unsqueeze(1).expand(-1, sents_embedding.size(0), -1)
            sents_expanded = sents_embedding.unsqueeze(0).expand(question_embeddings.size(0), -1, -1)
            # Compute the cosine similarity
            similarity_matrix = F.cosine_similarity(question_expanded, sents_expanded, dim=2)
            average_similarity = similarity_matrix.mean().item()
            similarity_scores.append(average_similarity)
        return similarity_scores[true_sent_idx], torch.tensor(similarity_scores)


    def forward(self, encoded_sents_ls, true_sent_idx, encoded_text, encoded_question, text_start, text_end, question_start, question_end, answer_start, answer_end):
        # Get the last hidden states for the text and the question
        text_embeddings = self.model(**encoded_text).last_hidden_state
        question_embeddings = self.model(**encoded_question).last_hidden_state

        sents_embedding_ls = []
        for sents in encoded_sents_ls:
            sent_embedding = self.model(**sents).last_hidden_state
            sents_embedding_ls.append(sent_embedding)

        # Calculate the similarity of the coreference spans
        coreference_similarity = coreference_cos_similarity(text_embeddings, question_embeddings, text_start, text_end, question_start, question_end, self.hidden_size)
        # Use F1 score to measure the correct answer span
        f1_score, _ = Model.cal_answer_f1(self, question_embeddings, text_embeddings, answer_start, answer_end)
        # Calculate the similarity between the question and the sents
        sents_similarity, _ = Model.sents_cos_similarity(self, question_embeddings, sents_embedding_ls, true_sent_idx)

        # Harmonize the cosine similarities and the F1 score
        scores = (coreference_similarity + 1) / 2 + f1_score + (sents_similarity + 1) / 2
        return torch.tensor(scores, requires_grad=True) # This is what needs to be maximized in the training data.

    def tokenize(self, s):
        return self.tokenizer(s, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens = True)
    
    def detokenize(self, nums):
        nums_decoded = self.tokenizer.decode(nums)
        return nums_decoded
    
    def find_most_similar_span(self, encoded_text, encoded_question, question_start, question_end, window_size = 0):
        question_embeddings = self.model(**encoded_question).last_hidden_state
        span_embeddings = question_embeddings[0, question_start:question_end].squeeze(0)
        if span_embeddings.size(0) == self.hidden_size:
            span_embeddings = span_embeddings.unsqueeze(0)
        text_embeddings = self.model(**encoded_text).last_hidden_state
        text_embeddings = text_embeddings[:, 1:-1]

        encoded_text = encoded_text['input_ids'][0, 1:-1]

        window_size = span_embeddings.size(0) + window_size
        max_similarity = 0
        most_similar_span = None
        most_similar_span_pos = None

        for i in range(text_embeddings.size(1) - window_size + 1):
            similarity = coreference_cos_similarity(text_embeddings, question_embeddings, i, i + window_size, question_start, question_end, self.hidden_size)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_span = encoded_text[i:i + window_size]
                most_similar_span_pos = (i, i + window_size)
        return self.detokenize(most_similar_span), most_similar_span_pos
    
    def find_answer(self, encoded_text, encoded_question, answer_start, answer_end):
        text_embeddings = self.model(**encoded_text).last_hidden_state
        question_embeddings = self.model(**encoded_question).last_hidden_state

        f1_score, pos = Model.cal_answer_f1(self, question_embeddings, text_embeddings, answer_start, answer_end)
        start_index = pos[0]
        end_index = pos[1]

        span_pos = None

        if end_index <= start_index:
            return 0, "Cannot find a valid answer for this question.", (0, 1)
        else:
            span_pos = (start_index, end_index)
            encoded_text = encoded_text['input_ids'][0, 1:-1]
            pred_answer = encoded_text[start_index:end_index]
            return f1_score, self.detokenize(pred_answer), span_pos
    
    def find_sentence(self, encoded_question, encoded_sents_ls, true_sent_idx):
        question_embeddings = self.model(**encoded_question).last_hidden_state
        sents_embedding_ls = []
        for sents in encoded_sents_ls:
            sent_embedding = self.model(**sents).last_hidden_state
            sents_embedding_ls.append(sent_embedding)
        _, similarity_scores = Model.sents_cos_similarity(self, question_embeddings, sents_embedding_ls, true_sent_idx)
        pred_idx = similarity_scores.argmax().item()
        if pred_idx == true_sent_idx:
            return 1
        else:
            return 0




def find_start_end_pos(encoded, encoded_span):
    encoded = encoded['input_ids'][0, :]
    # Delete the first and last token for the span
    encoded_span = encoded_span['input_ids'][0, 1:-1]

    len_encoded_span = encoded_span.size(0)
    for i in range(encoded.size(0) - len_encoded_span + 1):
        if torch.equal(encoded[i:i+len_encoded_span], encoded_span):
            return i, i + len_encoded_span
    return -1, -1


def coreference_cos_similarity(A, B, text_start, text_end, question_start, question_end, hidden_size):
    A = A[0, text_start:text_end]
    if A.size(0) == hidden_size:
        A = A.unsqueeze(0)

    B = B[0, question_start:question_end]
    if B.size(0) == hidden_size:
        B = B.unsqueeze(0)

    A_expanded = A.unsqueeze(1).expand(-1, B.size(0), -1)
    B_expanded = B.unsqueeze(0).expand(A.size(0), -1, -1)

    # Compute the cosine similarity
    similarity_matrix = F.cosine_similarity(A_expanded, B_expanded, dim=2)

    # Aggregate the results
    average_similarity = similarity_matrix.mean().item()
    return average_similarity