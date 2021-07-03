from typing import List, Tuple

import spacy
from spacy.training import offsets_to_biluo_tags

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import numpy
from utils import load_data

# Load the custom model
nlp = spacy.load("models/ner_brainzilla_puzzles_model_50_lg")
# nlp = spacy.load("en_core_web_sm")

# Load the testing data
docs = load_data("testing_data/brainzilla_testing_puzzles_15_adnotated.json")


def get_cleaned_label(label: str):
    """ Removes the BILOU tag from the entities
    e.g. transforms 'U-PERSON' in 'PERSON'"""
    if "-" in label:
        return label.split("-")[1]
    else:
        return label


def create_total_target_vector(processed_entities: List[str], given_label: str) -> List[str]:
    target_vector = []
    for clue_entities in processed_entities:
        # print(clue_entities)
        clue_text = nlp.make_doc(clue_entities[0])
        # entities = clue_entities[1]["entities"]
        entities = [x for x in clue_entities[1]["entities"] if x[2] == given_label]
        bilou_entities = offsets_to_biluo_tags(clue_text, entities)
        result = []
        for item in bilou_entities:
            result.append(get_cleaned_label(item))
        target_vector.extend(result)
    return target_vector


def create_total_target_vector_overall(processed_entities: List[str]) -> List[str]:
    target_vector = []
    for clue_entities in processed_entities:
        # print(clue_entities)
        clue_text = nlp.make_doc(clue_entities[0])
        # entities = clue_entities[1]["entities"]
        entities = [x for x in clue_entities[1]["entities"]]
        bilou_entities = offsets_to_biluo_tags(clue_text, entities)
        result = []
        for item in bilou_entities:
            result.append(get_cleaned_label(item))
        target_vector.extend(result)
    return target_vector


def create_prediction_vector(text, given_label):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text, given_label)]


def create_prediction_vector_overall(text):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions_overall(text)]


def create_total_prediction_vector(docs: list, given_label: str):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0], given_label))
    return prediction_vector


def create_total_prediction_vector_overall(docs: list):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector_overall(doc[0]))
    return prediction_vector


def get_all_ner_predictions(text, given_label):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ == given_label]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities



def get_all_ner_predictions_overall(text):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def get_dataset_labels(given_label: str):
    """Lists all the unique entities found in the testing data."""
    return sorted(set(create_total_target_vector(docs, given_label)))


def generate_confusion_matrix(processed_entities: List[str], given_label: str):
    classes = sorted(set(create_total_target_vector(processed_entities, given_label)))
    y_true = create_total_target_vector(processed_entities, given_label)
    y_pred = create_total_prediction_vector(processed_entities, given_label)
    # print(y_true)
    # print(y_pred)
    return confusion_matrix(y_true, y_pred, classes)


def generate_metrics(processed_entities: List[str], given_label: str) -> Tuple[float, float, float]:
    y_true = create_total_target_vector(processed_entities, given_label)
    y_pred = create_total_prediction_vector(processed_entities, given_label)
    precision = precision_score(y_true, y_pred, average='binary', pos_label=given_label)
    recall = recall_score(y_true, y_pred, average='binary', pos_label=given_label)
    score = f1_score(y_true, y_pred, average='binary', pos_label=given_label)

    print(y_true)
    print(y_pred)

    print("LABEL: ", given_label)
    print('Precision: %.3f\n' % precision +
          'Recall: %.3f\n' % recall +
          'F1 Score: %.3f\n' % score)
    return precision, recall, score


def plot_confusion_matrix(docs, given_label: str, normalize=False, cmap=pyplot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion Matrix, for SpaCy NER'
    classes = get_dataset_labels(given_label)

    # Compute confusion matrix
    cm = generate_confusion_matrix(docs, given_label)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return cm, ax, pyplot


def generate_metrics_overall(processed_entities: List[str]) -> Tuple[float, float, float]:
    y_true = create_total_target_vector_overall(processed_entities)
    y_pred = create_total_prediction_vector_overall(processed_entities)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    score = f1_score(y_true, y_pred, average=None)

    print(y_true)
    print(y_pred)

    print('Precision: %.3f\n' % precision +
          'Recall: %.3f\n' % recall +
          'F1 Score: %.3f\n' % score)
    return precision, recall, score


def main():
    # print(docs[0][0])
    # print(sorted(set(create_total_target_vector(docs, "PERSON"))))
    # generate_confusion_matrix(docs)
    # plot_confusion_matrix(docs, given_label='PERSON', normalize=False)
    # plot_confusion_matrix(docs, given_label='COLOR', normalize=False)
    #
    entities = ['CARDINAL', 'GPE', 'COLOR', 'DATE', 'PERSON', 'TIME', 'PROFESSION', 'HOBBY', 'CATEGORY', 'PRODUCT'
    'NORP', 'ANIMAL', 'QUANTITY', 'FRUIT', 'WORK_OF_ART']

    precision = []
    recall = []
    score = []
    for e in entities:
        prec, rec, sc = generate_metrics(docs, given_label=e)
        precision.append(prec)
        recall.append(rec)
        score.append(sc)
    mean_precision = numpy.mean(precision)
    mean_recall = numpy.mean(recall)
    mean_score = numpy.mean(score)

    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean Score: {mean_score}")
    # generate_metrics(docs, given_label="CARDINAL")
    # generate_metrics(docs, given_label="CATEGORY")
    # generate_metrics(docs, given_label="COLOR")
    # generate_metrics(docs, given_label="DATE")
    # generate_metrics(docs, given_label="FRUIT")
    # generate_metrics(docs, given_label="GPE")
    # generate_metrics(docs, given_label="HOBBY")
    # # generate_metrics(docs, given_label="LOC")
    # generate_metrics(docs, given_label="NORP")
    # # generate_metrics(docs, given_label="ORG")
    # generate_metrics(docs, given_label="PERSON")
    # generate_metrics(docs, given_label="PRODUCT")
    # generate_metrics(docs, given_label="PROFESSION")
    # generate_metrics(docs, given_label="QUANTITY")
    # generate_metrics(docs, given_label="TIME")
    # generate_metrics(docs, given_label="WORK_OF_ART")


    # (0.54, 0.4426229508196721, 0.48648648648648646)  - EN model
    # (1.0, 1.0, 1.0)  - Custom


if __name__ == '__main__':
    main()
