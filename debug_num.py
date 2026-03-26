import sys; sys.path.insert(0,'.')
from q2.english_detector import tag_english_words, get_english_words
import re

text = "हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे"
tagged = tag_english_words(text)
print("tagged:", repr(tagged))
result = re.findall(r'\[EN\](.+?)\[/EN\]', tagged)
print("direct regex:", result)
words = get_english_words(text)
print("get_english_words:", words)
