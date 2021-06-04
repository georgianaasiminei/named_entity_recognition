import random
from typing import List, Tuple

import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.training.example import Example

from repository.puzzle_repository import get_puzzle, get_puzzles_in_interval
from utils import load_data, save_data


def _build_patterns_list(file: str, type_: str) -> List[dict]:
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

    persons_patterns = _build_patterns_list("entity_rules/persons.json", "PERSON")
    ruler.add_patterns(persons_patterns)

    fruits_patterns = _build_patterns_list("entity_rules/fruits.json", "FRUIT")
    ruler.add_patterns(fruits_patterns)

    products_patterns = _build_patterns_list("entity_rules/products.json", "PRODUCT")
    ruler.add_patterns(products_patterns)

    animals_patterns = _build_patterns_list("entity_rules/animals.json", "ANIMAL")
    ruler.add_patterns(animals_patterns)

    colors_patterns = _build_patterns_list("entity_rules/colors.json", "COLOR")
    ruler.add_patterns(colors_patterns)

    activities_patterns = _build_patterns_list("entity_rules/activities.json", "ACTIVITY")
    ruler.add_patterns(activities_patterns)

    gpes_patterns = _build_patterns_list("entity_rules/gpes.json", "GPE")
    ruler.add_patterns(gpes_patterns)

    orgs_patterns = _build_patterns_list("entity_rules/orgs.json", "ORG")
    ruler.add_patterns(orgs_patterns)

    norp_patterns = _build_patterns_list("entity_rules/nationalities.json", "NORP")
    ruler.add_patterns(norp_patterns)

    domains_patterns = _build_patterns_list("entity_rules/domains.json", "DOMAIN")
    ruler.add_patterns(domains_patterns)

    categories_patterns = _build_patterns_list("entity_rules/categories.json", "CATEGORY")
    ruler.add_patterns(categories_patterns)

    dates_patterns = _build_patterns_list("entity_rules/dates.json", "DATE")
    ruler.add_patterns(dates_patterns)

    times_patterns = _build_patterns_list("entity_rules/times.json", "TIME")
    ruler.add_patterns(times_patterns)

    # save the model containing the `entity_ruler` pipe
    nlp.to_disk("puzzle_ner")


def create_training_set_for_a_puzzle(untrained_nlp, text: str) -> List[Tuple]:
    """will return a data structure as in TRAIN_DATA list
    Applies NER on a text (clues of a puzzle) and returns the text and all the found entities and their positions
    e.g.:
    TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    """
    doc = untrained_nlp(text)

    entities = []
    result = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.label_))
    if entities:
        result = [text, {"entities": entities}]
    return result


def create_train_data_file(untrained_nlp, clues_list: List[str]):
    TRAIN_DATA = []
    for clue in clues_list:
        ner_clue = create_training_set_for_a_puzzle(untrained_nlp, clue)
        if ner_clue:
            TRAIN_DATA.append(ner_clue)

    print(len(TRAIN_DATA))
    save_data("training_data/first_10_puzzles.json", TRAIN_DATA)


def pretty_print_ner(doc: str):
    """
    This will display nicely NER at this address  http://0.0.0.0:5001
    """

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
    try:
        displacy.serve(doc,
                       style="ent",
                       host="localhost",
                       port=5001,
                       options=options)
    except Exception as e:
        print(e)


def train_spacy(nlp, data, iterations):
    TRAIN_DATA = data

    ner = nlp.get_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner" and pipe != "entity_ruler"]
    with nlp.disable_pipes(*other_pipes):
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


def update_trained_model_with_new_rules():
    # RUN THIS if you've updated the ANNOTATED data (entity_rules/*.json)
    # This will create a new model called `puzzle_ner` and will save it to disk
    # Then we will create an enhanced nlp pipe from the original one by adding the custom rules
    generate_rules()
    untrained_custom_nlp = spacy.load("puzzle_ner")
    untrained_nlp = spacy.load("en_core_web_sm")
    untrained_nlp.add_pipe("entity_ruler", source=untrained_custom_nlp, before="ner")

    # Generates a Train DATA file with the first 10 clues and the found entities
    clues_list = get_puzzles_in_interval(1, 10)
    # clues_list.extend(get_puzzles_in_interval(61, 69))
    clues_text = [texts for _, texts in clues_list]
    create_train_data_file(untrained_nlp, clues_text)

    TRAIN_DATA = load_data("training_data/first_10_puzzles.json")
    # print(TRAIN_DATA)

    # TRAIN data and create a new nlp model
    result_nlp = train_spacy(untrained_nlp, TRAIN_DATA, 30)
    result_nlp.to_disk("ner_first_10_puzzles_model")


def test_model(input_puzzle: str, model: str):
    """
    This will run the trained model over a given input and will display the result at http://0.0.0.0:5001
    :return:
    """
    trained_nlp = spacy.load(model)
    # nlp = spacy.load("en_core_web_sm")
    doc = trained_nlp(input_puzzle)

    # This will display nicely NER at this address  http://0.0.0.0:5001
    pretty_print_ner(doc)


def main():
    # Comment it if you did not update the entity rules
    # update_trained_model_with_new_rules()

    # Test the model
    test = get_puzzle(13)
    print(test)
    test_model(test, "ner_first_10_puzzles_model")


if __name__ == '__main__':
    main()
