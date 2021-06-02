from typing import List, Tuple, Any

import spacy

from custom_training import pretty_print_ner
from repository.puzzle_repository import get_puzzles_in_interval
from utils import save_data

nlp = spacy.load("ner_first_10_puzzles_model")


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


def create_testing_data_file(clues_list: List[str]):
    docs = []
    TESTING_DATA = []
    for clue in clues_list:
        doc, ner_clue = create_testing_set_for_a_puzzle(clue)
        docs.append(doc)
        if ner_clue:
            TESTING_DATA.append(ner_clue)

    print(len(TESTING_DATA))
    pretty_print_ner(docs)
    save_data("testing_data/puzzles_from_40_to_50.json", TESTING_DATA)


def main():
    # Generates a Train DATA file with the first 10 clues and the found entities
    clues_list = get_puzzles_in_interval(41, 50)
    # for title, clue_text in clues_list:
    #     generate_output_file(title, clue_text)

    title, clue_text = clues_list[8]

    # res = extract_entities_from_clues(clue)
    # print(res)

    # create_testing_data_file(clues_list)


if __name__ == '__main__':
    main()