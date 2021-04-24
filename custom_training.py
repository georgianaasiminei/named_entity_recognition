import spacy
from spacy.lang.en import English
from spacy import displacy
import json

from repository.puzzle_repository import get_puzzle


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


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

    # save the model
    nlp.to_disk("puzzle_ner")


def create_training_set(text):
    """will return a data structure as in TRAIN_DATA list"""
    # TRAIN_DATA = [(text, {"entities": [(start, end, label)]})]
    nlp = spacy.load("puzzle_ner")
    doc = nlp(text)
    entities = []
    result = []
    for ent in doc.ents:
        entities.append((ent.start_char, ent.end_char, ent.text, ent.label_))
    if entities:
        result = [text, {"entities": entities}]

    print(result)


def main():
    # test_rules_on_puzzle()
    # create_training_set(text)

    # main_nlp = spacy.blank("en")
    # english_ner = custom_nlp.get_pipe("entity_ruler")
    # nlp.to_disk("en_core_web_sm_demo")

    # Create an enhanced nlp pipe from the original one by adding the custom rules
    custom_nlp = spacy.load("puzzle_ner")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("entity_ruler", source=custom_nlp, before="ner")

    generate_rules()
    text = get_puzzle(8)  # einstein
    doc = nlp(text)

    # pick custom colors for each entity
    colors = {"ANIMAL": "#778f8c",
              "ACTIVITY": "#f54242",
              "NORP": "#966c88",
              "FRUIT": "linear-gradient(90deg, #ecf542, #4287f5)",
              "COLOR": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {
        "ents": ["ACTIVITY", "ANIMAL", "CARDINAL", "COLOR", "DATE", "EVENT", "FAC", "FRUIT", "GPE", "LANGUAGE", "LAW",
                 "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
                 "WORK_OF_ART"],
        "colors": colors}
    displacy.serve(doc, style="ent", port=5001, options=options)


if __name__ == '__main__':
    main()
