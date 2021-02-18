import re
from collections import Counter
from typing import List

import en_core_web_sm
import requests
from bs4 import BeautifulSoup
from spacy import displacy

from repository.database import create_database
from repository.models import Puzzle, Source

nlp = en_core_web_sm.load()

DOMAIN = "https://www.brainzilla.com"
ZEBRA_PUZZLES_PATH = "/logic/zebra/"
SOURCE_ID = 1


def extract_pages() -> List[str]:
    start_url = f"{DOMAIN}{ZEBRA_PUZZLES_PATH}"
    html_doc = requests.get(start_url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    pages_paths = soup.find('div', class_="col-lg-8").findAll('li')
    pages = []
    for page in pages_paths:
        pages.append(page.a['href'])
    return pages


def extract_puzzle(puzzle_url: str) -> Puzzle:
    html_doc = requests.get(puzzle_url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    title_selector = soup.find('div', class_="page-header").find('h1').text
    title = re.search("(.*)\sZebra", title_selector).group(1)
    description = soup.find('div', class_="page-header").find('div', class_='description').text
    clues = soup.find('div', class_="clues").text
    return Puzzle(
        title=title.strip(),
        description=description.strip(),
        clues=clues.strip(),
        url=puzzle_url,
        source_id=SOURCE_ID
    )


def populate_db(puzzles: List[Puzzle]):
    # Add Source
    brainzilla_source = Source.create(
        name="Brainzilla",
        domain="https://www.brainzilla.com",
        puzzles_path="/logic/zebra/"
    )
    brainzilla_source.save()

    # Add puzzles
    for puzzle in puzzles:
        puzzle.save()


def main():
    # Extract the puzzle
    puzzle_url = "https://www.brainzilla.com/logic/zebra/blood-donation/"
    print("#################################################")
    puzzle_text = extract_puzzle(puzzle_url)
    print(puzzle_text)
    print("#################################################")

    pages_urls = extract_pages()
    print(pages_urls)

    # Extract all puzzles
    puzzles = []
    for path in pages_urls:
        full_path = f"{DOMAIN}{path}"
        extracted_puzzle = extract_puzzle(full_path)
        puzzles.append(extracted_puzzle)
        # print(extracted_puzzle.title, extracted_puzzle.url, extracted_puzzle.description, extracted_puzzle.clues)
        print(extracted_puzzle.title)

    # Initiate DB
    create_database()
    # populate_db(puzzles)

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
