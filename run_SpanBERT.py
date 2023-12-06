import torch
import random
import copy
import sys
import utils
import SpanBERT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)

def train(train_data, dev_data):
    train_dict = utils.load_data(train_data)
    dev_dict = utils.load_data(dev_data)

    model = SpanBERT.Model(hidden_size=1024, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0003)

    best_dev_loss = None
    train_coreference_f1 = 0.
    train_answer_f1 = 0.
    train_sents_acc = 0.
    train_counts = 0
    for epoch in range(20):
        model.train()
        # shuffle the train_dict
        keys = list(train_dict.keys())
        random.shuffle(keys)
        train_data = {key: train_dict[key] for key in keys}

        train_loss = 0.
        for example_id in train_data:
            ex = train_data[example_id]
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
            
            scores = model.forward(encoded_sents_ls, true_sent_idx, encoded_text, encoded_question, text_start, text_end, question_start, question_end, answer_start, answer_end)
            loss = -scores
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

            _, coreference_pos = model.find_most_similar_span(encoded_text, encoded_question, question_start, question_end)
            coreference_f1_score = utils.cal_f1_scores(coreference_pos[0], coreference_pos[1], text_start, text_end)
            train_coreference_f1 += coreference_f1_score

            answer_f1_score, _, _ = model.find_answer(encoded_text, encoded_question, answer_start, answer_end)
            train_answer_f1 += answer_f1_score

            sents_val = model.find_sentence(encoded_question, encoded_sents_ls, true_sent_idx)
            train_sents_acc += sents_val

            train_counts += 1
        
        ave_train_coreference_f1 = train_coreference_f1/train_counts
        ave_train_answer_f1 = train_answer_f1/train_counts
        ave_train_sents_acc = train_sents_acc/train_counts
        print(f'train_coreference_f1={ave_train_coreference_f1} train_answer_f1={ave_train_answer_f1} train_sents_acc={ave_train_sents_acc}', file=sys.stderr, flush=True)

        dev_loss = 0.
        dev_coreference_f1 = 0.
        dev_answer_f1 = 0.
        dev_sents_acc = 0.
        dev_counts = 0
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

            scores = model.forward(encoded_sents_ls, true_sent_idx, encoded_text, encoded_question, text_start, text_end, question_start, question_end, answer_start, answer_end)
            loss = -scores
            dev_loss += loss.item()
            _, coreference_pos = model.find_most_similar_span(encoded_text, encoded_question, question_start, question_end)
            coreference_f1_score = utils.cal_f1_scores(coreference_pos[0], coreference_pos[1], text_start, text_end)
            dev_coreference_f1 += coreference_f1_score

            answer_f1_score, _, _ = model.find_answer(encoded_text, encoded_question, answer_start, answer_end)
            dev_answer_f1 += answer_f1_score

            sents_val = model.find_sentence(encoded_question, encoded_sents_ls, true_sent_idx)
            dev_sents_acc += sents_val

            dev_counts += 1

        ave_dev_coreference_f1 = dev_coreference_f1/dev_counts
        ave_dev_answer_f1 = dev_answer_f1/dev_counts
        ave_dev_sents_acc = dev_sents_acc/dev_counts
        print(f'dev_coreference_f1={ave_dev_coreference_f1} dev_answer_f1={ave_dev_answer_f1} dev_sents_acc={ave_dev_sents_acc}', file=sys.stderr, flush=True)

        if best_dev_loss is None or dev_loss < best_dev_loss:
            best_model = copy.deepcopy(model)
            best_dev_loss = dev_loss
        else:
            # When the learning rate gets too low, keep learning rate unchanged
            if opt.param_groups[0]['lr'] < 1e-5:
               opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']
            else: 
                # If dev_loss didn't improve, halve the learning rate
                opt.param_groups[0]['lr'] *= 0.5
            print(f"lr={opt.param_groups[0]['lr']}")
        
        print(f'[{epoch+1}] train_loss={train_loss} dev_loss={dev_loss}', file=sys.stderr, flush=True)
    return best_model



if __name__ == "__main__":
    train_data = './Data/qed-train.jsonlines'
    dev_data = './Data/qed-dev.jsonlines'
    model = train(train_data, dev_data)

    torch.save(model, './best_model.pt')
