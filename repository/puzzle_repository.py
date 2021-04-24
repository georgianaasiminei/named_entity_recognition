from repository.models import Puzzle


def get_puzzle(uid: int) -> str:
    """Return the clues for a given Puzzle ID"""
    result = Puzzle.get(Puzzle.id == uid)
    if not result:
        raise Exception(f"There is no puzzle with ID {uid}")
    return result.clues


def get_puzzles_in_interval(from_uid: int, to_uid: int) -> str:
    """Return the clues for a given Puzzle ID"""
    retrieved_puzzles = list(Puzzle.filter(Puzzle.id >= from_uid, Puzzle.id <= to_uid))
    if not retrieved_puzzles:
        raise Exception(f"There is no puzzle with ID {from_uid}")
    clues = [puzzle.clues for puzzle in retrieved_puzzles]
    result = "\n\n".join(clues)
    return result
