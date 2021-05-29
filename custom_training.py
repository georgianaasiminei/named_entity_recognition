import random
from typing import List, Tuple

import spacy
from spacy.lang.en import English
from spacy import displacy

from spacy.training.example import Example

from repository.puzzle_repository import get_puzzle, get_puzzles_in_interval

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
from utils import load_data, save_data


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
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if entities:
        result = [text, {"entities": entities}]
    return result


def create_train_data_file(clues_list: List[str]):
    TRAIN_DATA = []
    for clue in clues_list:
        ner_clue = create_training_set_for_a_puzzle(clue)
        if ner_clue:
            TRAIN_DATA.append(ner_clue)

    print(len(TRAIN_DATA))
    save_data("training_data/first_10_puzzles.json", TRAIN_DATA)


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
    displacy.serve(doc,
                   style="ent",
                   port=5001,
                   options=options)


def train_spacy(data, iterations):
    TRAIN_DATA = data
    custom_nlp = spacy.load("puzzle_ner")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")
    ner = nlp.get_pipe("ner")

    # nlp = spacy.blank("en")
    # if "ner" not in nlp.pipe_names:
    #     ner = nlp.create_pipe("ner")
    #     nlp.add_pipe("ner", last=True)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner" and pipe != "entity_ruler"]
    with nlp.disable_pipes(*other_pipes):
        # optimizer = nlp.begin_training()
        optimizer = nlp.create_optimizer()
        for itn in range(iterations):
            print(f"Starting iteration {str(itn)}")
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example],
                           drop=0.2,
                           sgd=optimizer,
                           losses=losses
                           )
            print(losses)
    return nlp


def main():
    # test_rules_on_puzzle()

    # main_nlp = spacy.blank("en")
    # english_ner = custom_nlp.get_pipe("entity_ruler")
    # nlp.to_disk("en_core_web_sm_demo")

    # RUN THIS if you've updated the ANNOTATED data (data/*.json)
    # This will create a new model called `puzzle_ner` and will save it to disk
    # generate_rules()

    # Create an enhanced nlp pipe from the original one by adding the custom rules
    # custom_nlp = spacy.load("puzzle_ner")
    # nlp = spacy.load("en_core_web_sm")
    # nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")

    # text = get_puzzle(20)  # einstein
    # Generates a Train DATA file with the first 10 clues and the found entities
    # clues_list = get_puzzles_in_interval(1, 10)
    # create_train_data_file(clues_list)

    # TRAIN_DATA = load_data("training_data/first_10_puzzles.json")
    # print(TRAIN_DATA)

    # TRAIN data and create a new nlp model
    # nlp = train_spacy(TRAIN_DATA, 30)
    # nlp.to_disk("ner_first_10_puzzles_model")

    # Test the model
    test = get_puzzle(35)  # einstein
    print(test)
    nlp = spacy.load("ner_first_10_puzzles_model")
    # nlp = spacy.load("en_core_web_sm")
    doc = nlp(test)
    # for ent in doc.ents:
    #     print(ent.text, ent.label_)

    # This will display nicely NER at this address  http://0.0.0.0:5001
    # doc = nlp(text)

    pretty_print_ner(doc)


if __name__ == '__main__':
    main()
