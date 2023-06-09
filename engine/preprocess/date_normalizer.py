from datetime import datetime
from typing import List
from dateutil import parser
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class DateNormalizer:
    def __init__(self):
        pass

    def normalize_date(words: List[str]) -> str:
        new_texts = []
        for word in words:
            new_text = word
            try:
                dt = parser.parse(word)
                if isinstance(dt, datetime):
                    date_obj = parser.parse(word)
                    formatted_date = date_obj.strftime("%Y %m %d")
                    day_name = date_obj.strftime("%A")
                    month_name = date_obj.strftime("%B")
                    time_obj = date_obj.time().strftime("%I %M %p")
                    new_formatted = f"{formatted_date} {day_name} {month_name} {time_obj}"
                    new_text = new_text.replace(word, new_formatted)
            except (ValueError, OverflowError):
                pass
            new_texts.append(new_text)
        return " ".join(new_texts)


