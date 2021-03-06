from collections import defaultdict
from pprint import pprint
from typing import List

import spacy

from repository.puzzle_repository import get_puzzles_in_interval, get_puzzle_with_title, get_testing_puzzles, \
    get_testing_puzzles_with_title

nlp = spacy.load("models/ner_brainzilla_puzzles_model_50_lg_final")


def extract_entities_from_clues(clue_text: str) -> dict:
    """
    Generates a nested dict from a given clue text:
    {
        'PERSON': {
            'Alvin': [(637, 642), (986, 991), (993, 998)],
            'Brenda': [(888, 894)],
            },
        'COLOR': {
            'Black': [(95, 100)],
            'Blue': [(337, 341)],
            },
        ...
    }
    """
    doc = nlp(clue_text)

    result = defaultdict(lambda: defaultdict(list))

    for ent in doc.ents:
        result[ent.label_][ent.text].append((ent.start_char, ent.end_char))
    return result


def _write_to_file(data: List[List], output_file: str):
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("set(arithmetic).\nassign(domain_size, 5).\nassign(max_models, -1).\nlist(distinct).\n")
        f.writelines([f"{row}.\n" for row in data])
        f.write("end_of_list.\n")
    print(f"Wrote to file: {output_file}\n")


def generate_output_file(clue_title: str, clue_text: str) -> List[List]:
    """Generates an output file for the mace4 input."""
    entities = extract_entities_from_clues(clue_text)
    pprint(dict(entities))
    result = [list(filter(lambda x: x != "one", ent_text.keys()))  # eliminate `one` as NE
              for ent_label, ent_text in entities.items()
              if ent_label != "ORDINAL"]  # eliminate ordinals as NE

    result = [x for x in result if x]  # eliminate empty lists



    # Optional - can be deleted
    # rest = []
    # result2 = []
    # for ents in result:
    #     if len(ents) <= 2:  # if the list has 2 or less elements, it is merged with another one
    #         rest.extend(ents)
    #     else:
    #         result2.append(ents)
    # result2.append(rest)

    file_name = f"./mace4_files/{clue_title.lower().replace(' ', '_')}_output"
    _write_to_file(
        sorted(result, key=len, reverse=True),
        file_name)

    return result


def main():
    # clues_list = get_puzzles_in_interval(41, 50)
    # for title, clue_text in clues_list:
    #     generate_output_file(title, clue_text)

    # puzzle_title, puzzle_clues = get_puzzle_with_title(11)
    # generate_output_file(clue_title=puzzle_title, clue_text=puzzle_clues)

    testing_puzzles_ids = [11, 13, 17, 19, 27, 33, 36, 42, 47, 51, 55, 58, 61, 64, 69]
    puzzles_list = get_testing_puzzles_with_title(testing_puzzles_ids)
    for puzzle_title, puzzle_clues in puzzles_list:
        generate_output_file(clue_title=puzzle_title, clue_text=puzzle_clues)

    # title, clue_text = clues_list[8]
    # generate_output_file(title, clue_text)


if __name__ == '__main__':
    main()
