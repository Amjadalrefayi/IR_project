import enchant
from typing import List


class SpellCheck:

    def __init__(self) -> None:
        self.spell_checker = enchant.Dict("en_US")

    def correct(self, words: List[str]) -> str:
        return " ".join([self.spell_checker.suggest(token)[0] if not self.spell_checker.check(
            token) and self.spell_checker.suggest(token) else token
                         for token in words])
