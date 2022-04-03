    
# from dataclasses import dataclass
# from collections import namedtuple

def Word(text, upos):
    return text, upos

# @dataclass
# class Word:
#     text: str = None
#     upos: str = None
#     # head: Word = None
    
#     def __hash__(self):
#         return hash((self.text, self.upos))
#     def __str__(self):
#         return self.text
#     def __repr__(self):
#         return f'{self.text} ({self.upos})'