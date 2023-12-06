import torch
import sys
sys.path.append('/Users/david/Desktop/CSE60657/Project/NLP_project/')
# sys.path.append('/afs/crc.nd.edu/user/d/dgan/CSE60657/NLP_project')
#import layers
#import seqlabel
import SpanBERT
import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load("./best_model.pt", map_location=torch.device(device))

dev_data = './Data/qed-dev.jsonlines'
dev_dict = utils.load_data(dev_data)

coreference_f1 = 0.
answer_f1 = 0.
sents_acc = 0.
counts = 0

for example_id in dev_dict:
    ex = dev_dict[example_id]
    text = ex.passage
    question = ex.question
    coreference = ex.aligned_nps
    answer_ls = ex.answer
    selected_sent = ex.selected_sent
    sentence_starts = []

    if len(ex.sentence_starts) == 0:
        sentence_starts = [0, selected_sent['end']]
    else: 
        sentence_starts = ex.sentence_starts
    true_sent_idx = utils.get_true_sent_idx(selected_sent, sentence_starts)
    sents_ls = utils.separate_sents(text, sentence_starts)

    # Skip samples where there is no coreference
    if utils.check_coreference(coreference):
        continue
    elif len(coreference) >= 2: # When there are multiple coreferences, only use the first one
        coreference = [coreference[0]]

    assert coreference[0][0].type == 'question'
    assert coreference[0][1].type == 'context'
    question_span = utils.get_encoded_span(question, coreference, type=0)
    encoded_question_span = model.tokenize(question_span)
    encoded_question = model.tokenize(question)
    question_start, question_end = SpanBERT.find_start_end_pos(encoded_question, encoded_question_span)
    if question_start == -1 or question_end == -1: # If the start and end positions cannot be found, skip.
        continue

    text_span = utils.get_encoded_span(text, coreference, type=1)
    encoded_text_span = model.tokenize(text_span)
    encoded_text = model.tokenize(text)
    text_start, text_end = SpanBERT.find_start_end_pos(encoded_text, encoded_text_span)
    if text_start == -1 or text_end == -1: # If the start and end positions cannot be found, skip.
        continue

    answer_span = utils.get_answer_span(text, answer_ls)
    encoded_answer_span = model.tokenize(answer_span)
    answer_start, answer_end = SpanBERT.find_start_end_pos(encoded_text, encoded_answer_span)
    if answer_start == -1 or answer_end == -1: # If the start and end positions cannot be found, skip.
        continue

    encoded_sents_ls = []
    for sents in sents_ls:
        encoded_sent = model.tokenize(sents)
        encoded_sents_ls.append(encoded_sent)
    
    coreference_span, coreference_pos = utils.find_most_similar_span(model, encoded_text, encoded_question, question_start, question_end)
    coreference_f1_score = utils.cal_f1_scores(coreference_pos[0], coreference_pos[1], text_start, text_end)
    coreference_f1 += coreference_f1_score

    answer_f1_score, answer_span, answer_pos = utils.find_answer(model, encoded_text, encoded_question, answer_start, answer_end)
    answer_f1 += answer_f1_score

    sents_val = utils.find_sentence(model, encoded_question, encoded_sents_ls, true_sent_idx)
    sents_acc += sents_val

    counts += 1
    print(sents_acc/counts)


ave_coreference_f1 = coreference_f1/counts
ave_answer_f1 = answer_f1/counts
ave_sents_acc = sents_acc/counts

print('Final output----')
print(ave_coreference_f1)
print(ave_answer_f1)
print(ave_sents_acc)