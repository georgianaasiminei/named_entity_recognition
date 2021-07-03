import random
from typing import List, Tuple

import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.training.example import Example

from repository.puzzle_repository import get_puzzle, get_puzzles_in_interval, get_training_puzzles, get_testing_puzzles
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

    professions_patterns = _build_patterns_list("entity_rules/professions.json", "PROFESSION")
    ruler.add_patterns(professions_patterns)

    gpes_patterns = _build_patterns_list("entity_rules/gpes.json", "GPE")
    ruler.add_patterns(gpes_patterns)

    orgs_patterns = _build_patterns_list("entity_rules/orgs.json", "ORG")
    ruler.add_patterns(orgs_patterns)

    norp_patterns = _build_patterns_list("entity_rules/nationalities.json", "NORP")
    ruler.add_patterns(norp_patterns)

    hobbies_patterns = _build_patterns_list("entity_rules/hobbies.json", "HOBBY")
    ruler.add_patterns(hobbies_patterns)

    categories_patterns = _build_patterns_list("entity_rules/categories.json", "CATEGORY")
    ruler.add_patterns(categories_patterns)

    locations_patterns = _build_patterns_list("entity_rules/locations.json", "LOC")
    ruler.add_patterns(locations_patterns)

    dates_patterns = _build_patterns_list("entity_rules/dates.json", "DATE")
    ruler.add_patterns(dates_patterns)

    times_patterns = _build_patterns_list("entity_rules/times.json", "TIME")
    ruler.add_patterns(times_patterns)

    cardinals_patterns = _build_patterns_list("entity_rules/cardinals.json", "CARDINAL")
    ruler.add_patterns(cardinals_patterns)

    quantities_patterns = _build_patterns_list("entity_rules/quantities.json", "QUANTITY")
    ruler.add_patterns(quantities_patterns)

    woa_patterns = _build_patterns_list("entity_rules/work_of_art.json", "WORK_OF_ART")
    ruler.add_patterns(woa_patterns)

    # save the model containing the `entity_ruler` pipe
    nlp.to_disk("models/puzzle_ner")


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


def create_train_data_file(untrained_nlp, clues_list: List[str], file_name):
    TRAIN_DATA = []
    for clue in clues_list:
        ner_clue = create_training_set_for_a_puzzle(untrained_nlp, clue)
        if ner_clue:
            TRAIN_DATA.append(ner_clue)

    print(len(TRAIN_DATA))
    save_data(file_name, TRAIN_DATA)


def pretty_print_ner(doc: str):
    """
    This will display nicely NER at this address  http://0.0.0.0:5001
    """

    # pick custom colors for each entity
    colors = {"ANIMAL": "#778f8c",
              "PROFESSION": "#f54242",
              "NORP": "#966c88",
              "HOBBY": "#fcba03",
              "CATEGORY": "#85cbf2",
              # "OBJECT": "#960f55",
              "FRUIT": "linear-gradient(90deg, #ecf542, #4287f5)",
              "COLOR": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {
        "ents": ["ANIMAL", "CARDINAL", "CATEGORY", "COLOR", "DATE", "EVENT", "FAC", "FRUIT", "GPE", "HOBBY",
                 "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
                 "PRODUCT", "PROFESSION", "QUANTITY", "TIME", "WORK_OF_ART"],
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
    untrained_custom_nlp = spacy.load("models/puzzle_ner")
    # untrained_nlp = spacy.load("en_core_web_sm")
    untrained_nlp = spacy.load("en_core_web_lg")
    untrained_nlp.add_pipe("entity_ruler", source=untrained_custom_nlp, before="ner")

    # Generates a Train DATA file with the first 10 clues and the found entities
    # clues_list = get_puzzles_in_interval(1, 10)

    # Generates a Train DATA file with the 54 clues from brainzilla and the found entities
    testing_puzzles_ids = [11, 13, 17, 19, 27, 33, 36, 42, 47, 51, 55, 58, 61, 64, 69]
    clues_list = get_training_puzzles(excluded_ids=testing_puzzles_ids)

    # clues_list.extend(get_puzzles_in_interval(61, 69))
    # clues_text = [texts for _, texts in clues_list]
    training_data_file = "training_data/54_brainzilla_puzzles.json"
    create_train_data_file(untrained_nlp, clues_list, training_data_file)

    TRAIN_DATA = load_data("training_data/54_brainzilla_puzzles_annotated.json")
    # print(TRAIN_DATA)

    # TRAIN data and create a new nlp model
    result_nlp = train_spacy(untrained_nlp, TRAIN_DATA, 50)
    result_nlp.to_disk("models/ner_brainzilla_puzzles_model_50_lg_final")


def test_model(input_puzzle: str, model: str):
    """
    This will run the trained model over a given input and will display the result at http://0.0.0.0:5001
    :return:
    """
    trained_nlp = spacy.load(model)
    doc = trained_nlp(input_puzzle)
    entities = [(x.text, x.label_) for x in doc.ents]
    print(entities)

    # This will display nicely NER at this address  http://0.0.0.0:5001
    pretty_print_ner(doc)


def test_model_on_testing_list(model: str):
    testing_puzzles_ids = [11, 13, 17, 19, 27, 33, 36, 42, 47, 51, 55, 58, 61, 64, 69]
    clues_list = get_testing_puzzles(ids=testing_puzzles_ids)
    docs = []
    trained_nlp = spacy.load(model)
    for clue in clues_list:
        doc = trained_nlp(clue)
        docs.append(doc)
    return docs


def generate_testing_data_file(model):
    trained_nlp = spacy.load(model)
    testing_puzzles_ids = [11, 13, 17, 19, 27, 33, 36, 42, 47, 51, 55, 58, 61, 64, 69]
    clues_list = get_testing_puzzles(testing_puzzles_ids)
    create_train_data_file(trained_nlp, clues_list, "testing_data/testing_15_brainzilla_puzzles_unannotated.json")


def main():
    # Comment it if you did not update the entity rules
    # update_trained_model_with_new_rules()

    # Test the model
    # test = get_puzzle(41)
    # example = "•  Jackie is next to the person who eats on the lounge.\n •  Arnotts brand cookies are kept in a round jar.\n •  The person beside Cameron eats cookies at a table.\n •  The person who eats Oreos eats in the closet\n •  Julieanne likes Paradise brand cookies\n •  The person who drinks banana milk is in the middle and owns a tall jar\n •  The first person likes vanilla milk\n •  Holly is the person on the far right\n •  The person who eats in the bedroom drinks strawberry milk\n •  The person who owns the tall jar is next to the person who owns square jar\n •  Cameron drinks caramel milk\n •  The person who likes the Dick Smith brand is next to the person who likes the Coles brand\n •  The person who likes the No Frills brand is next to the person who owns a round jar\n •  The person who stole the 100s and 1000s cookies is next to the person who owns the brass jar\n •  The second person from the right eats No Frills brand and is next to the person who owns a round jar\n •  The first person on the left stole the choc chip cookies\n •  The person who eats Dick Smith brand is next to the person who eats Paradise brand\n •  The second from the left has a brass jar\n •  Julieanne is to the right of the person who drinks strawberry milk\n •  The person who drinks chocolate milk does it at the table\n •  The Paradise brand cookies are eaten in the kitchen\n •  The person who eats Tiny Teddies doesn't keep them in a round jar\n •  The Coles brand cookies are kept in a mini sized jar\n\nA Ginger Cookie was also stolen."
    example = """1. Daniella Black and her husband work as Shop-Assistants.
2. The book "The Seadog" was brought by a couple who drive a Fiat and love the color red.
3. Owen and his wife Victoria like the color brown.
4. Stan Horricks and his wife Hannah like the color white.
5. Jenny Smith and her husband work as Warehouse Managers and they drive a Wartburg.
6. Monica and her husband Alexander borrowed the book "Grandfather Joseph".
7. Mathew and his wife like the color pink and brought the book "Mulatka Gabriela".
8. Irene and her husband Oto work as Accountants.
9. The book "We Were Five" was borrowed by a couple driving a Trabant.
10. The Cermaks are both Ticket-Collectors who brought the book "Shed Stoat".
11. Mr and Mrs Kuril are both Doctors who borrowed the book "Slovacko Judge".
12. Paul and his wife like the color green.
13. Veronica Dvorak and her husband like the color blue.
14. Rick and his wife brought the book "Slovacko Judge" and they drive a Ziguli.
15. One couple brought the book "Dame Commissar" and borrowed the book "Mulatka Gabriela".
16. The couple who drive a Dacia, love the color violet.
17. The couple who work as Teachers borrowed the book "Dame Commissar".
18. The couple who work as Agriculturalists drive a Moskvic.
19. Pamela and her husband drive a Renault and brought the book "Grandfather Joseph".
20. Pamela and her husband borrowed the book that Mr and Mrs Zajac brought.
21. Robert and his wife like the color yellow and borrowed the book "The Modern Comedy".
22. Mr and Mrs Swain work as Shoppers.
23. "The Modern Comedy" was brought by a couple driving a Skoda."""
    test_model(example, "models/ner_brainzilla_puzzles_model_50_lg_final")
    # print(test)

    # Generate a starting file for the testing data to be corrected manually
    # generate_testing_data_file("models/ner_brainzilla_puzzles_model_50_lg")

    # docs = test_model_on_testing_list("models/ner_brainzilla_puzzles_model_50_lg")
    # docs = test_model_on_testing_list("models/ner_brainzilla_puzzles_model_30_lg")
    # pretty_print_ner(docs)


if __name__ == '__main__':
    main()
