from repository.models import Puzzle


def get_puzzle(uid: int) -> str:
    """Return the clues for a given Puzzle ID"""
    result = Puzzle.get(Puzzle.id == uid)
    if not result:
        raise Exception(f"There is no puzzle with ID {uid}")
    return result.clues
