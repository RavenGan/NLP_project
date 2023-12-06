import utils
import TFIDF_mod
from tqdm import tqdm

train_dat = utils.load_data("./Data/qed-train.jsonlines")
dev_dat = utils.load_data("./Data/qed-dev.jsonlines")
def train(dat_dict):
    n = len(dat_dict)
    counts = 0
    for example_id in tqdm(dat_dict, desc="Processing"):
        example = dat_dict[example_id]
        
        # Check whether there is only one sentence in the passage
        sentence_starts = []
        if len(example.sentence_starts) == 0:
            sentence_starts = [0, example.selected_sent['end']]
        else: 
            sentence_starts = example.sentence_starts

        pred = TFIDF_mod.tfidf_qa(example.passage, example.question, sentence_starts)
        true_idx = TFIDF_mod.get_true_sent_idx(example.selected_sent, sentence_starts)
        counts += TFIDF_mod.criterion(true_idx, pred)
    acc = counts/n
    return(acc)

train_acc = train(train_dat)
dev_acc = train(dev_dat)

print(f'Training accuracy={train_acc}') # 0.5927
print(f'Dev accuracy={dev_acc}') # 0.5612