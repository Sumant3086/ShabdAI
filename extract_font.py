from fontTools.ttLib import TTCollection
import os

ttc = TTCollection('C:/Windows/Fonts/Nirmala.ttc')
for i, font in enumerate(ttc.fonts):
    name = font['name'].getDebugName(4)
    print(i, name)
    out = f'Nirmala_{i}.ttf'
    font.save(out)
    print(f'  saved -> {out} ({os.path.getsize(out)//1024} KB)')
