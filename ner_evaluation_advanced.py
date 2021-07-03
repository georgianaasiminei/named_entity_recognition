from collections import defaultdict, namedtuple
from copy import deepcopy
from pprint import pprint

import nereval
import spacy
# from nereval import Entity
from ner_eval import compute_metrics, compute_precision_recall_wrapper
from utils import load_data



# Load the final model
nlp = spacy.load("models/ner_brainzilla_puzzles_model_50_lg")

# Load the testing data
annotated_data = load_data("testing_data/brainzilla_testing_puzzles_15_adnotated.json")


example = "The customer whose delivery time is 25 days is somewhere between the customer whose delivery time is 20 days and the customer whose delivery time is 10 days, in that order.\nLori is next to the youngest woman.\nAt the fourth position is the 45 years old customer.\nThe customer who bought the most expensive piece of furniture is next to the customer whose delivery will take 5 days.\nThe woman wearing the Yellow shirt is somewhere between the woman who bought the $900 piece of furniture and the 40-year-old woman, in that order.\nThe customer who purchased the $900 piece of furniture is next to the customer whose delivery time is 20 days.\nThe woman wearing the Green shirt is somewhere between the woman who bought the Table and the woman wearing the Red shirt, in that order.\nThe woman wearing the Orange shirt is somewhere to the right of the woman wearing the Red shirt.\nDana is somewhere between the customer who bought the Wardrobe and Lori, in that order.\nThe woman wearing the Green shirt is exactly to the left of the woman whose delivery time is 10 days.\nBarbara is next to the customer who bought the Wardrobe.\nThe woman whose delivery time is 25 days is somewhere between the woman wearing the Yellow shirt and the woman whose delivery time is 5 days, in that order.\nThe 40 years old customer is next to the customer who purchased the $1100 furniture.\nAt the first position is the woman who bought the Table.\nThe customer who purchased the $1100 piece of furniture is next to the customer who purchased the $800 piece of furniture.\nThe Cupboard was bought by the customer that is somewhere between Barbara and the 45 years old customer, in that order.\nThe 40-year-old woman is next to the 45-year-old woman.\nThe oldest customer is wearing the Yellow shirt.\nPatricia is somewhere between the woman who bought the $900 piece of furniture and the woman whose delivery will take 25 days, in that order.\nThe customer that purchased the Dresser is next to the customer wearing the Green shirt.\nThe 50-year-old woman is next to the woman wearing the Yellow shirt."

testing_data = [data[0] for data in annotated_data]
true_entities = [data[1]["entities"][0] for data in annotated_data]


def build_prediction_vector(data: list, given_label=None):
    y_pred = []
    for text in data:
        doc = nlp(text)
        for ent in doc.ents:
            new_ent = Entity(ent.orth_, ent.label_, ent.start_char)
            y_pred.append(new_ent)
    # y_pred.append(Entity(ent.orth_, ent.label_, ent.start_char) for ent in doc.ents)
    print(len(y_pred))
    return y_pred


def build_prediction_vector_by_entity(data: list, given_label=None):
    by_entity = defaultdict(list)
    for text in data:
        doc = nlp(text)
        for ent in doc.ents:
            new_ent = Entity(ent.orth_, ent.label_, ent.start_char)
            by_entity[ent.label_].append(new_ent)
    # y_pred.append(Entity(ent.orth_, ent.label_, ent.start_char) for ent in doc.ents)
    return by_entity


example = annotated_data[0]


def build_true_vector(data: list):
    y_true = []
    for puzzle in data:
        text, entities = puzzle
        # print(text)
        # print(entities)
        for start, end, label in entities["entities"]:
            new_ent = Entity(text[start:end], label, start)
            y_true.append(new_ent)
    print(len(y_true), y_true)
    return y_true


def build_true_vector_by_entity(data: list):
    by_entity = defaultdict(list)
    for puzzle in data:
        text, entities = puzzle
        # print(text)
        # print(entities)
        for start, end, label in entities["entities"]:
            new_ent = Entity(text[start:end], label, start)
            by_entity[label].append(new_ent)
    return by_entity


# y_pred = build_prediction_vector(testing_data)
# y_true = build_true_vector(annotated_data)
#
# score = nereval.evaluate([y_true], [y_pred])
# print('F1-score: %.2f' % score)
#
#
# y_pred_by_ent = build_prediction_vector_by_entity(testing_data)
# y_true_by_ent = build_true_vector_by_entity(annotated_data)
# print(len(y_pred_by_ent))
# print(len(y_true_by_ent))
# print(set(y_pred_by_ent.keys()) - set(y_true_by_ent.keys()))
# for ent in y_pred_by_ent:
#     f1_score = nereval.evaluate([y_true_by_ent.get(ent, [])], [y_pred_by_ent[ent]])
#     print(f"{ent}: {f1_score}")


Entity = namedtuple("Entity", "e_type start_offset end_offset")


def build_prediction_vector_2(data: list):
    y_pred = []
    for text in data:
        doc = nlp(text)
        for ent in doc.ents:
            new_ent = Entity(e_type=ent.label_, start_offset=ent.start_char, end_offset=ent.end_char)
            y_pred.append(new_ent)
    print(len(y_pred), y_pred)
    return y_pred


def build_true_vector_2(data: list):
    y_true = []
    for puzzle in data:
        text, entities = puzzle
        # print(text)
        # print(entities)
        for start, end, label in entities["entities"]:
            new_ent = Entity(text[start:end], label, start)
            new_ent = Entity(e_type=label, start_offset=start, end_offset=end)
            y_true.append(new_ent)
    print(len(y_true), y_true)
    return y_true


pred = build_prediction_vector_2(testing_data)
true = build_true_vector_2(annotated_data)

y_true_by_ent = build_true_vector_by_entity(annotated_data)
entity_list = y_true_by_ent.keys()
print(entity_list)


metrics_results = {'correct': 0, 'incorrect': 0, 'partial': 0,
                   'missed': 0, 'spurious': 0, 'possible': 0, 'actual': 0, 'precision': 0, 'recall': 0}

# overall results
results = {'strict': deepcopy(metrics_results),
           'ent_type': deepcopy(metrics_results),
           'partial': deepcopy(metrics_results),
           'exact': deepcopy(metrics_results)
          }

evaluation_agg_entities_type = {e: deepcopy(results) for e in entity_list}



print(len(pred))
print(len(true))
tmp_results, tmp_agg_results = compute_metrics(true, pred, entity_list)

for eval_schema in results.keys():
    for metric in metrics_results.keys():
        results[eval_schema][metric] += tmp_results[eval_schema][metric]

results = compute_precision_recall_wrapper(results)

for e_type in entity_list:

    for eval_schema in tmp_agg_results[e_type]:

        for metric in tmp_agg_results[e_type][eval_schema]:
            evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

    # Calculate precision recall at the individual entity level

    evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(evaluation_agg_entities_type[e_type])


lines = ['ent_type', 'exact', 'partial', 'strict']
columns = ['actual', 'correct', 'incorrect', 'missed', 'partial', 'possible', 'precision', 'recall', 'spurious']
from csv import writer

# with open('marci_results.csv', 'w', newline="") as f:
#     csv = writer(f, delimiter=',')
#     csv.writerow(["Measure"] + [x.title() for x in lines])
#     for column in columns:
#         res = [column.title()]
#         for line in lines:
#             res.append(results[line][column])
#         csv.writerow(res)
#     csv.writerow([])
#
#     for entity in evaluation_agg_entities_type:
#         csv.writerow([entity])
#         csv.writerow([])
#         csv.writerow(["Measure"] + [x.title() for x in lines])
#         for column in columns:
#             res = [column.title()]
#             for line in lines:
#                 res.append(evaluation_agg_entities_type[entity][line][column])
#             csv.writerow(res)
#         csv.writerow([])



#pprint(results)
#pprint(evaluation_agg_entities_type)