from typing import List, Tuple

from repository.models import Puzzle


def get_puzzle(uid: int) -> str:
    """Return the clues for a given Puzzle ID"""
    result = Puzzle.get(Puzzle.id == uid)
    if not result:
        raise Exception(f"There is no puzzle with ID {uid}")
    return result.clues


def get_puzzle_with_title(uid: int) -> Tuple[str, str]:
    """Return the clues for a given Puzzle ID"""
    puzzle = Puzzle.get(Puzzle.id == uid)
    if not puzzle:
        raise Exception(f"There is no puzzle with ID {uid}")
    return puzzle.title, puzzle.clues


def get_puzzles_in_interval(from_uid: int, to_uid: int) -> List:
    """Return the clues for a given interval of Puzzle ID"""
    retrieved_puzzles = list(Puzzle.filter(Puzzle.id >= from_uid, Puzzle.id <= to_uid))
    if not retrieved_puzzles:
        raise Exception(f"There is no puzzle with ID {from_uid}")
    clues = [(puzzle.title, puzzle.clues) for puzzle in retrieved_puzzles]
    # result = "\n\n".join(clues)
    return clues


def get_testing_puzzles(ids: List[int]) -> List[str]:
    """Returns the clues of the puzzles having the id in the given ids list"""
    all_puzzles = Puzzle.select()
    puzzles_dict = {puzzle.id: puzzle.clues for puzzle in all_puzzles}
    clues = [clue for id_, clue in puzzles_dict.items() if id_ in ids]
    return clues


def get_training_puzzles(excluded_ids: List[int]) -> List[str]:
    """Returns the clues of the puzzles having the id in the given ids list"""
    all_puzzles = Puzzle.select()
    puzzles_dict = {puzzle.id: puzzle.clues for puzzle in all_puzzles}
    clues = [clue for id_, clue in puzzles_dict.items() if id_ not in excluded_ids and id_ < 70]
    return clues

