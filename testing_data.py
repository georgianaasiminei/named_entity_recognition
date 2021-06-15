import re
from typing import List, Tuple, Any

import spacy

from custom_training import pretty_print_ner
from repository.puzzle_repository import get_puzzles_in_interval, get_testing_puzzles, get_training_puzzles
from utils import save_data

# nlp = spacy.load("ner_first_10_puzzles_model")


def create_testing_set_for_a_puzzle(text: str) -> Tuple[Any, List[Tuple]]:
    """will return a data structure as in TRAIN_DATA list
    Applies NER on a text (clues of a puzzle) and returns the text and all the found entities and their positions

    TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    """

    # Create an enhanced nlp pipe from the original one by adding the custom rules
    # custom_nlp = spacy.load("puzzle_ner")
    # nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")
    #
    # doc = nlp(text)
    doc = nlp(text)

    entities = []
    result = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if entities:
        result = [text, {"entities": entities}]
    return doc, result


def create_testing_data_file(clues_list: List[str], file_name: str):
    docs = []
    TESTING_DATA = []
    for clue in clues_list:
        doc, ner_clue = create_testing_set_for_a_puzzle(clue)
        docs.append(doc)
        if ner_clue:
            TESTING_DATA.append(ner_clue)

    print(len(TESTING_DATA))
    save_data(f"testing_data/{file_name}", TESTING_DATA)
    pretty_print_ner(docs)
    # save_data("testing_data/puzzles_from_40_to_50.json", TESTING_DATA)


def get_entity_coordinates(entity_text: str, clue_text: str) -> Tuple[str, List[Tuple]]:
    """Returns the start and end indexes of all the occurrences of a substring in a string"""
    entitity_coordinates_matches = re.finditer(entity_text, clue_text)
    entitity_coordinates = [(match.start(), match.end()) for match in entitity_coordinates_matches]
    return entity_text, entitity_coordinates


def ner_on_list(clues_list: List[str]):
    nlp = spacy.load("models/ner_brainzilla_puzzles_model_50_lg")
    docs = []
    for text in clues_list:
        doc = nlp(text)
        docs.append(doc)
    pretty_print_ner(docs)


def main():
    # Generates a Train DATA file with the first 10 clues and the found entities

    # title, clue_text = clues_list[8]

    # res = extract_entities_from_clues(clue)
    # print(res)

    testing_puzzles_ids = [11, 13, 17, 19, 27, 33, 36, 42, 47, 51, 55, 58, 61, 64, 69]
    # clues_list = get_testing_puzzles(ids=testing_puzzles_ids)
    clues_list = get_training_puzzles(excluded_ids=testing_puzzles_ids)
    print(len(clues_list))
    # ner_on_list(clues_list)

    # create_testing_data_file(clues_list, "brainzilla_testing_puzzles_15.json")

    print(get_entity_coordinates("30", clues_list[53]))


if __name__ == '__main__':
    main()
