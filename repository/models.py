from peewee import *


db = SqliteDatabase('puzzles.db')


class BaseModel(Model):
    class Meta:
        database = db


class Source(BaseModel):
    name = CharField()
    domain = CharField()
    puzzles_path = CharField()


class Puzzle(BaseModel):
    title = CharField()
    text = TextField()
    clues = TextField()
    url = CharField()
    source = ForeignKeyField(Source)
