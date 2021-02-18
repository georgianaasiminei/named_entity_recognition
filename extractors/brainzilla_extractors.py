from collections import Counter
from typing import List

import en_core_web_sm
import requests
from bs4 import BeautifulSoup
from spacy import displacy

nlp = en_core_web_sm.load()


def extract_brainzilla_pages() -> List[str]:
    braizilla_url = "https://www.brainzilla.com/logic/zebra/"
    html_doc = requests.get(braizilla_url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    pages_paths = soup.find('div', class_="col-lg-8").findAll('li')
    pages = []
    for page in pages_paths:
        pages.append(page.a['href'])
    return pages


def extract_puzzle(puzzle_url: str) -> str:
    html_doc = requests.get(puzzle_url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    puzzle_clues = soup.find('div', class_="clues")
    return puzzle_clues.text


def main():
    # Extract the puzzle's text
    puzzle_url = "https://www.brainzilla.com/logic/zebra/blood-donation/"
    print("#################################################")
    puzzle_text = extract_puzzle(puzzle_url)
    print(puzzle_text)
    print("#################################################")

    print(extract_brainzilla_pages())

    # # Apply NLP
    # processed_puzzle = nlp(puzzle_text)
    # displacy.render(processed_puzzle, jupyter=True, style='ent')
    #
    # # Extract NER
    # print("Labels of found entities:")
    # labels = [x.label_ for x in processed_puzzle.ents]
    # print(Counter(labels))
    #
    # print("Found entities by their types:")
    # entities = [(x.text, x.label_) for x in processed_puzzle.ents]
    # print(Counter(entities))


if __name__ == '__main__':
    main()
