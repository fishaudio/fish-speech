import json
import locale
from pathlib import Path

I18N_FILE_PATH = Path(__file__).parent / "locale"
DEFAULT_LANGUAGE = "en_US"


def load_language_list(language):
    with open(I18N_FILE_PATH / f"{language}.json", "r", encoding="utf-8") as f:
        language_list = json.load(f)

    return language_list


class I18nAuto:
    def __init__(self):
        i18n_file = Path(".locale")

        if i18n_file.exists():
            with open(i18n_file, "r", encoding="utf-8") as f:
                language = f.read().strip()
        else:
            # getlocale can't identify the system's language ((None, None))
            language = locale.getdefaultlocale()[0]

        if (I18N_FILE_PATH / f"{language}.json").exists() is False:
            language = DEFAULT_LANGUAGE

        self.language = language
        self.language_map = load_language_list(language)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return "Use Language: " + self.language


i18n = I18nAuto()
