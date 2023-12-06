import json
import re
import string
import attr
from absl import logging
from typing import Any, List, Mapping, Text, Tuple
from SpanBERT import *

def normalize_text(text: Text) -> Text:
  """Lowercases text and removes punctuation, articles and extra whitespace."""

  # Replace a, an, the with a space
  def remove_articles(s):
    return re.sub(r'\b(a|an|the)\b', ' ', s)
  # Remove punctuation
  def replace_punctuation(s):
    to_replace = set(string.punctuation)
    return ''.join('' if ch in to_replace else ch for ch in s)
  # Remove extra whitespace
  def white_space_fix(s):
    return ' '.join(s.split())

  text = text.lower() # Lowercase
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)
  return text

@attr.s(frozen=True)
class Entity:
  """Entity in either document or query."""

  # Inclusive start char offset of this entity mention. -1 refers to the start
  # of the answering sentence. The answering sentence is given in the data
  # as example["annotation"]["selected_sentence"].
  start_offset = attr.ib(type=int)

  # Exclusive end char offset of this entity mention. -1 refers to the entire
  # answering sentence.
  end_offset = attr.ib(type=int)
  # type must be either context or query.
  type = attr.ib(type=Text)
  # entity mention text.
  text = attr.ib(type=Text)
  normalized_text = attr.ib(type=Text)

  def __hash__(self):
    return hash((self.start_offset, self.end_offset, self.type))

  def __eq__(self, other):
    return (self.start_offset == other.start_offset and
            self.end_offset == other.end_offset and self.type == other.type)

@attr.s
class QEDExample:
  """A single training/test example."""
  example_id = attr.ib(type=int)
  title = attr.ib(type=Text)
  question = attr.ib(type=Text)
  passage = attr.ib(type=Text)
  sentence_starts = attr.ib(type=list)
  selected_sent = attr.ib(type=dict)
  answer = attr.ib(type=List[Entity])
  nq_answers = attr.ib(type=List[List[Entity]])
  # the first entity is query entity, the second is document entity.
  aligned_nps = attr.ib(type=List[Tuple[Entity, Entity]])
  # either single_sentence or multi_sentence.
  explanation_type = attr.ib(type=Text)


def load_answer(answer: List[Mapping[Text, Any]]) -> List[Entity]:
  """Loads annotated QED answer, potentially composed of multiple spans."""
  output_answer = []
  for a in answer:
    output_answer.append(
        Entity(
            text=a['paragraph_reference']['string'],
            normalized_text=normalize_text(a['paragraph_reference']['string']),
            start_offset=a['paragraph_reference']['start'],
            end_offset=a['paragraph_reference']['end'],
            type='context'))
  return output_answer


def load_nq_answers(
    answer_list: List[List[Mapping[Text, Any]]]) -> List[List[Entity]]:
  """Loads annotated NQ answers, each potentially composed of multiple spans."""
  output_answer_list = []
  for answer in answer_list:
    output_answer = []
    for a in answer:
      output_answer.append(
          Entity(
              text=a['string'],
              normalized_text=normalize_text(a['string']),
              start_offset=a['start'],
              end_offset=a['end'],
              type='context'))
    output_answer_list.append(output_answer)
  return output_answer_list


def load_aligned_entities(alignment_dict: List[Mapping[Text, Any]],
                          question_text: Text,
                          context_text: Text) -> List[Tuple[Entity, Entity]]:
  """Loads aligned entities from json."""
  aligned_nps = []
  for single_np_alignment in alignment_dict:
    q_entity_text = single_np_alignment['question_reference']['string']
    q_entity_offset = (single_np_alignment['question_reference']['start'],
                       single_np_alignment['question_reference']['end'])
    c_entity_text = single_np_alignment['sentence_reference']['string']
    c_entity_offset = (single_np_alignment['sentence_reference']['start'],
                       single_np_alignment['sentence_reference']['end'])
    if q_entity_text != question_text[q_entity_offset[0]:q_entity_offset[1]]:
      logging.error(
          'Question entity offset not proper. from text: %s, from byte offset %s',
          q_entity_text, question_text[q_entity_offset[0]:q_entity_offset[1]])
      raise ValueError()

    question_entity = Entity(
        text=question_text[q_entity_offset[0]:q_entity_offset[1]],
        normalized_text=normalize_text(q_entity_text),
        start_offset=q_entity_offset[0],
        end_offset=q_entity_offset[1],
        type='question')
    if c_entity_offset[0] != -1:
      if c_entity_text != context_text[c_entity_offset[0]:c_entity_offset[1]]:
        logging.error(
            'Context entity offset not proper. from text: %s, from byte offset %s',
            c_entity_text, context_text[c_entity_offset[0]:c_entity_offset[1]])
        raise ValueError()
      doc_entity = Entity(
          text=context_text[c_entity_offset[0]:c_entity_offset[1]],
          normalized_text=normalize_text(c_entity_text),
          start_offset=c_entity_offset[0],
          end_offset=c_entity_offset[1],
          type='context')
    else:  # this is a bridging linguistic context instance.
      doc_entity = Entity(
          text='',
          start_offset=-1,
          end_offset=-1,
          type='context',
          normalized_text='')
    aligned_nps.append((question_entity, doc_entity))
  return aligned_nps


def load_single_line(elem: Mapping[Text, Any]) -> QEDExample:
  """Loads a QEDExample from json."""
  return QEDExample(
      example_id=elem['example_id'],
      title=elem['title_text'],
      question=elem['question_text'],
      passage=elem['paragraph_text'],
      sentence_starts=elem['sentence_starts'],
      selected_sent=elem['annotation'].get('selected_sentence', []),
      answer=load_answer(elem['annotation'].get('answer', [])),
      nq_answers=load_nq_answers(elem['original_nq_answers']),
      aligned_nps=load_aligned_entities(
          elem['annotation'].get('referential_equalities', []),
          elem['question_text'],
          elem['paragraph_text']),
      explanation_type=elem['annotation']['explanation_type'])


def load_data(fname: Text) -> Mapping[int, QEDExample]:
  """Loads jsonl data and outputs a dict mapping example_id to QEDExample."""
  output_dict = {}
  incorrectly_formatted = 0
  with open(fname) as f:
    for line in f:
      try:
        elem = json.loads(line)
        example = load_single_line(elem)
        if example.explanation_type == 'single_sentence':
          output_dict[example.example_id] = example
      except ValueError:
        incorrectly_formatted += 1
  logging.info('%d examples not correctly formatted and skipped.',
               incorrectly_formatted)
  return output_dict


def get_encoded_span(sents, coreference, type=0): # 0 for 'question'; 1 for 'context'
    start = coreference[0][type].start_offset
    end = coreference[0][type].end_offset
    span = list(sents)[start:end]
    span = ''.join(span)
    return span

def check_coreference(coreference):
    if len(coreference) == 0: # If there is no coreference, skip
        return True
    elif coreference[0][0].start_offset <= 0 or coreference[0][0].end_offset <= 0:
        return True
    elif coreference[0][1].start_offset <= 0 or coreference[0][1].end_offset <= 0:
        return True
    
def cal_f1_scores(pred_start, pred_end, true_start, true_end):
  if pred_end <= pred_start:
    return 0
  elif true_end <= true_start:
    return 0
  else:
    pred_span = set(range(pred_start, pred_end))
    true_span = set(range(true_start, true_end))
    overlap = len(pred_span.intersection(true_span))
    precision = overlap / len(pred_span)
    recall = overlap / len(true_span)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score

def get_answer_span(sents, answer_ls):
  start = answer_ls[0].start_offset
  end = answer_ls[0].end_offset
  span = list(sents)[start:end]
  span = ''.join(span)
  return span

def separate_sents(text, sents_starts):
    # Separate a passage into sentences based on the starting indices.
    characters = list(text) # Split the passage into characters
    sents_ls = []
    for i in range(len(sents_starts)):
        # If it's the last index, slice until the end of the main list
        if i == len(sents_starts) - 1:
            sub_char = characters[sents_starts[i]:]
            sentence = ''.join(sub_char)
            sents_ls.append(sentence)
        else:
            # Slice from the current index to the next index
            sub_char = characters[sents_starts[i]:sents_starts[i+1]]
            sentence = ''.join(sub_char)
            sents_ls.append(sentence)
    return sents_ls

def get_true_sent_idx(selected_sent, sents_start):
    # Get the correct selected sentence index
    start_idx = selected_sent['start']
    true_sent_idx = sents_start.index(start_idx)
    return true_sent_idx
