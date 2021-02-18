from peewee import SqliteDatabase

from repository.models import Puzzle, Source


def create_database():
    db = SqliteDatabase('puzzles.db')

    db.connect()
    db.create_tables([Source, Puzzle])
    return db
