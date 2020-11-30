import json
from pathlib import Path
import re
from pprint import pp

here = Path(__file__).parent


pattern = re.compile('\[\"(.*?)\"\]')  # noqa : W605


def f():
    f = here / "Data-Engineer-Case" / "data" / "sales-2019-01-01.json"
    data = f.read_text()
    m = pattern.match(data)
    data = m.group(1)
    print(data)
    data = data.replace("\\\"", "\"")
    print(data)
    j = json.loads(data)
    pp(j)


if __name__ == "__main__":
    f()
