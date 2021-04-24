from typing import List, Tuple

import spacy
from spacy.lang.en import English
from spacy import displacy
import json

from repository.puzzle_repository import get_puzzle, get_puzzles_in_interval


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# def test_rules_on_puzzle():
#     # adnotam manual nationalitatile, citindu-le din fisier
#     norp_patterns = create_training_data("data/nationalities.json", "NORP")
#     colors_patterns = create_training_data("data/colors.json", "Color")
#     generate_rules(norp_patterns)
#     print(norp_patterns)
#     generate_rules(colors_patterns)
#     print(colors_patterns)
#     nlp = spacy.load("puzzle_ner")
#
#     puzzle_clues = get_puzzle(20)
#     print(puzzle_clues)
#     print(test_model(nlp, puzzle_clues))
#     doc = nlp(puzzle_clues)
#     # create a local server
#     # displacy.serve(doc, style="ent", port=5001)


# def test_model(nlp, text):
#     doc = nlp(text)
#     results = []
#     for ent in doc.ents:
#         results.append((ent.text, ent.label_))
#     return results


def _build_patterns_list(file: str, type_: str):
    data = load_data(file)
    patterns = []
    for item in data:
        pattern = {"label": type_,
                   "pattern": item}
        patterns.append(pattern)
    return patterns


def generate_rules():
    nlp = English()
    # add a pipe with rules based NER
    ruler = nlp.add_pipe("entity_ruler")

    persons_patterns = _build_patterns_list("data/persons.json", "PERSON")
    ruler.add_patterns(persons_patterns)

    fruits_patterns = _build_patterns_list("data/fruits.json", "FRUIT")
    ruler.add_patterns(fruits_patterns)

    products_patterns = _build_patterns_list("data/products.json", "PRODUCT")
    ruler.add_patterns(products_patterns)

    woa_patterns = _build_patterns_list("data/work_of_arts.json", "WORK_OF_ART")
    ruler.add_patterns(woa_patterns)

    animals_patterns = _build_patterns_list("data/animals.json", "ANIMAL")
    ruler.add_patterns(animals_patterns)

    colors_patterns = _build_patterns_list("data/colors.json", "COLOR")
    ruler.add_patterns(colors_patterns)

    activities_patterns = _build_patterns_list("data/activities.json", "ACTIVITY")
    ruler.add_patterns(activities_patterns)

    gpes_patterns = _build_patterns_list("data/gpes.json", "GPE")
    ruler.add_patterns(gpes_patterns)

    orgs_patterns = _build_patterns_list("data/orgs.json", "ORG")
    ruler.add_patterns(orgs_patterns)

    norp_patterns = _build_patterns_list("data/nationalities.json", "NORP")
    ruler.add_patterns(norp_patterns)

    domains_patterns = _build_patterns_list("data/domains.json", "DOMAIN")
    ruler.add_patterns(domains_patterns)

    categories_patterns = _build_patterns_list("data/categories.json", "CATEGORY")
    ruler.add_patterns(categories_patterns)

    dates_patterns = _build_patterns_list("data/dates.json", "DATE")
    ruler.add_patterns(dates_patterns)

    # save the model
    nlp.to_disk("puzzle_ner")


def create_training_set_for_a_puzzle(text: str) -> List[Tuple]:
    """will return a data structure as in TRAIN_DATA list
    Applies NER on a text (clues of a puzzle) and returns the text and all the found entities and their positions"""
    # TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]

    # Create an enhanced nlp pipe from the original one by adding the custom rules
    custom_nlp = spacy.load("puzzle_ner")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")

    doc = nlp(text)

    entities = []
    result = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.text, ent.label_))
    if entities:
        result = [text, {"entities": entities}]
    return result


def pretty_print_ner(doc: str):
    # pick custom colors for each entity
    colors = {"ANIMAL": "#778f8c",
              "ACTIVITY": "#f54242",
              "NORP": "#966c88",
              "DOMAIN": "#fcba03",
              "CATEGORY": "#85cbf2",
              # "NEW_COLOR": "#960f55",
              "FRUIT": "linear-gradient(90deg, #ecf542, #4287f5)",
              "COLOR": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {
        "ents": ["ACTIVITY", "ANIMAL", "CARDINAL", "CATEGORY", "COLOR", "DATE", "DOMAIN", "EVENT", "FAC", "FRUIT",
                 "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT",
                 "QUANTITY", "TIME", "WORK_OF_ART"],
        "colors": colors}
    displacy.serve(doc, style="ent", port=5001, options=options)


def main():
    # test_rules_on_puzzle()

    # main_nlp = spacy.blank("en")
    # english_ner = custom_nlp.get_pipe("entity_ruler")
    # nlp.to_disk("en_core_web_sm_demo")

    # RUN THIS if you've updated the ANNOTATED data (data/*.json)
    # This will create a new model called `puzzle_ner` and will save it to disk
    # generate_rules()

    # Create an enhanced nlp pipe from the original one by adding the custom rules
    custom_nlp = spacy.load("puzzle_ner")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")

    # text = get_puzzle(20)  # einstein
    # Generates a Train DATA file with the first 10 clues and the found entities
    clues_list = get_puzzles_in_interval(1, 10)
    TRAIN_DATA = []
    for clue in clues_list:
        ner_clue = create_training_set_for_a_puzzle(clue)
        if ner_clue:
            TRAIN_DATA.append(ner_clue)

    print(len(TRAIN_DATA))
    save_data("training_data/first_10_puzzles.json", TRAIN_DATA)

    # This will display nicely NER at this address  http://0.0.0.0:5001
    # doc = nlp(text)
    # pretty_print_ner(doc)


if __name__ == '__main__':
    main()
