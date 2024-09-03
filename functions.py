
import re, json, base64, io, matplotlib
from Dictionary.Word import Word
import xml.etree.ElementTree as ET
from turkish.deasciifier import Deasciifier
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trnlp import (
    TrnlpWord,
    syllabification,
    writeable,
    levenshtein_distance,
    word_token,
)



tr_analyzer = TrnlpWord()
sentiment_dataset = "resources\\Turkish_sentiliteralnet.xml"
wordnet_dataset = "resources\\Turkish_wordnet.xml"
turkish_words = frozenset(line.strip() for line in open("resources\\NEW_turkish_Words.txt"))
turkish_vowels = frozenset("aeıioöuüâî")


with open("resources\\Turkish_words_dict.json", "r", encoding="utf-8") as json_file:
    all_turkish_words_dict = json.load(json_file)


def deasciifier_function_FUNCTIONUSEAGE(text):
    deasciifier = Deasciifier(text.strip())
    return deasciifier.convert_to_turkish()


def negativity_value_FUNCTIONUSEAGE(word):
    global tr_analyzer
    tr_analyzer.setword(word)
    return tr_analyzer.is_negative()


def vowel_extractor_RHYMEGENERATOR(a_word):
    global turkish_vowels
    return tuple(char for char in a_word if char in turkish_vowels)


def syllable_harmony_RHYMEGENERATOR(a_word):
    vowel_instance = re.compile(r"[aeıioöuüâî]")
    consonant_instance = re.compile(r"[bcçdfgğhjklmnprsştvyzxqw]")
    input_syllables = syllabification(a_word)
    harmony = []

    for part in input_syllables:
        substitution = re.sub(vowel_instance, "V", part)
        substitution = re.sub(consonant_instance, "C", substitution)
        harmony.append(substitution)

    return tuple(harmony)


###########################################################################################################################################


def vowel_extractor(word):
    global turkish_vowels
    return "_".join([char for char in word if char in turkish_vowels])


def syllable_harmony(word):
    vowel_instance = re.compile(r"[aeıioöuüâî]")
    consonant_instance = re.compile(r"[bcçdfgğhjklmnprsştvyzxqw]")
    input_syllables = syllabification(word)
    harmony = []

    for part in input_syllables:
        substitution = re.sub(vowel_instance, "V", part)
        substitution = re.sub(consonant_instance, "C", substitution)
        harmony.append(substitution)

    return "".join(harmony)


def morphology_analysis(word):
    global tr_analyzer

    try:
        tr_analyzer.setword(word)
        return writeable(tr_analyzer.get_morphology, long=True)     
    except Exception:
        return None


def etymology_analysis(word):
    global tr_analyzer

    try:
        tr_analyzer.setword(word)
        trnlp_analysis = tr_analyzer.get_morphology
        return trnlp_analysis["etymon"]
    except Exception:
        return None


def word_type(word):
    global tr_analyzer

    try:
        tr_analyzer.setword(word)
        trnlp_analysis = tr_analyzer.get_morphologys
        return f"{word}: " + "/".join(trnlp_analysis["baseType"]) 
    except Exception:
        return None


def sound_event(word):
    global tr_analyzer

    try:
        tr_analyzer.setword(word)
        trnlp_analysis = tr_analyzer.get_morphology
        does_event_exist = trnlp_analysis["event"] == 1

        if does_event_exist:
            verdict_2 = "vardır."
        else:
            verdict_2 = "yoktur."

        return f"Kökte ses olayı {verdict_2}"
    except Exception:
        return None


def plurality_analysis(word):
    global tr_analyzer

    try:
        tr_analyzer.setword(word)
        plurality_test = tr_analyzer.is_plural()
        is_singular = plurality_test == 0

        if is_singular:
            return "Tekil İfade"
        else:
            return "Çoğul İfade"
    except Exception:
        return None


def wordnet_analysis(wordnet_dataset, literal_name):
    tree = ET.parse(wordnet_dataset)
    root = tree.getroot()

    for w in root.findall("SYNSET"):
        synonym = w.find("SYNONYM")
        if synonym is not None:
            for literal in synonym.findall("LITERAL"):
                if literal.text == literal_name:
                    if w.find("DEF").text != ' None ':
                        return f"{literal_name}: " + f"' {w.find("DEF").text} '"
    return None


def word_information(input_word):

    if Word.isPunctuationSymbol(input_word):
        return f" '{input_word}' : Noktalama işareti"
    elif Word.isHonorific(input_word):
        return f" '{input_word}' : Hitap sözcüğü"
    elif Word.isOrganization(input_word):
        return f" '{input_word}' : Organizasyon"
    elif Word.isMoney(input_word):
        return f" '{input_word}' : Para birimi"
    elif Word.isTime(input_word):
        return f" '{input_word}' : Zaman ifadesi"
    return None


###########################################################################################################################################


def rhyme_generator(input_word):
    global turkish_words
    global all_turkish_words_dict

    input_syllabification = tuple(syllabification(input_word))
    input_vowels = vowel_extractor_RHYMEGENERATOR(input_word)
    input_harmony = syllable_harmony_RHYMEGENERATOR(input_word)
    input_length = len(input_syllabification)

    filter_1 = {
        word
        for word in turkish_words
        if len(syllabification(word)) == input_length
        and levenshtein_distance(input_word, word) <= 5
    }

    filter_2 = set()

    for word in filter_1:
        if word in all_turkish_words_dict:
            dict_vowels = all_turkish_words_dict[word][0]
            dict_harmony = all_turkish_words_dict[word][1]
            vowel_score = sum(
                1
                for i in range(min(len(input_vowels), len(dict_vowels)))
                if input_vowels[i] == dict_vowels[i]
            )
            harmony_score = sum(
                1
                for i in range(min(len(input_harmony), len(dict_harmony)))
                if input_harmony[i] == dict_harmony[i]
            )
            if vowel_score == len(input_vowels) and harmony_score > 1:
                filter_2.add(word)

    return filter_2


def phonetic_analysis(word):

    phonetic_map = {
        "c": "dʒ",
        "y": "j",
        "v": "ʋ",
        "a": "α",
        "ş": "ʃ",
        "ç": "tʃ",
        "j": "ʒ",
        "ı": "ɯ",
        "ü": "y",
        "ö": "ø",
    }

    analysis = "".join((phonetic_map.get(i, i) for i in word))

    allophone_rules = [
        (r"l(?=[αɯou])|(?<=[αɯou])l", "ɫ"),
        (r"g(?=[eiøy])|(?<=[eiøy])g", "ɟ"),
        (r"(?<=[αeɯioøuy])ğ(?=[αeɯioøuy])", "•"),
        (r"(?<=[^e])ğ", ": "),
        (r"eğ(?=[^αeɯioøuy])", "ej"),
        (r"h(?=[eiøy])", "ç"),
        (r"(?<=[αɯou])h", "x"),
        (r"ʋ(?=[uyoø])|(?<=[uyoø])ʋ", "β"),
        (r"f(?=[uyoø])|(?<=[uyoø])f", "ɸ"),
        (r"k(?=[αɯou])", "kʰ"),
        (r"k(?=[eiøy])", "cʰ"),
        (r"(?<=[eiøy])n(?=[cɟcʰ])", "ɲ"),
        (r"(?<=[αɯou])n(?=[kgkʰ])", "ŋ"),
        (r"m(?=[fɸ])", "ɱ"),
        (r"p(?=[αeɯioøuy])", "pʰ"),
        (r"t(?=[αeɯioøuy])", "tʰ"),
        (r"α(?=[mɱnŋɲ])", "α̃ "),
        (r"(?<=[ɟlcʰ])α(?=[ɟlcʰ])", "a"),
        (r"o(?=[mɱnŋɲrɾ̥lɫ])", "ɔ"),
        (r"ø(?=[mɱnŋɲrɾ̥lɫ])", "œ"),
        (r"e(?=[mɱnŋɲrɾ̥lɫ])", "ɛ"),
        (r"h$", "ç"),
        (r"r$", "ɾ̥"),
        (r"(?<=[eiøy])k$", "c"),
        (r"o$", "ɔ"),
        (r"ø$", "œ"),
    ]

    for pattern, replacement in allophone_rules:
        analysis = re.sub(pattern, replacement, analysis)

    return f"[ {analysis} ]"


def word_sentiment_analysis(sentiment_dataset, word):
    tree = ET.parse(sentiment_dataset)
    root = tree.getroot()
    negativity = negativity_value_FUNCTIONUSEAGE(word)

    total_value = 0

    for w in root.findall("WORD"):
        title = w.find("NAME").text
        if title == word:
            n_score = float(w.find("NSCORE").text)
            p_score = float(w.find("PSCORE").text)

            if n_score == p_score:
                if negativity > 0:
                    total_value += -int(negativity)

                elif negativity == 0:
                    total_value += 0

            elif n_score > p_score:
                total_value += -int(
                    ((n_score + negativity) / 2))

            elif n_score < p_score:
                total_value += int(p_score)

        else:
            if negativity > 0:
                total_value += -int(negativity)
            elif negativity == 0:
                total_value += 0
    
    value_score = total_value*100

    if value_score > 100:
        return 100
    elif value_score < -100:
        return -100
    else:
        return value_score
    

def text_sentiment_analysis(sentiment_dataset, text):
    tree = ET.parse(sentiment_dataset)
    root = tree.getroot()

    total_value = 0
    count = 0

    normalized_text = deasciifier_function_FUNCTIONUSEAGE(text)
    text_1 = word_token(normalized_text)

    for i in text_1:
        negativity = negativity_value_FUNCTIONUSEAGE(i)

        for w in root.findall("WORD"):
            title = w.find("NAME").text
            if title == i:
                n_score = float(w.find("NSCORE").text)
                p_score = float(w.find("PSCORE").text)

                if n_score == p_score:
                    if negativity > 0:
                        count += 1
                        total_value += -negativity
                    elif negativity == 0:
                        total_value += 0

                elif n_score > p_score:
                    count += 1
                    total_value += (-n_score - negativity) / 2

                elif n_score < p_score:
                    count += 1
                    total_value += p_score

            else:
                if negativity > 0:
                    count += 1
                    total_value += -negativity
                elif negativity == 0:
                    total_value += 0

    mean_value = total_value / count if count != 0 else 0
    senti_score = int(mean_value * 100)

    if senti_score > 100:
        return 100
    elif senti_score < -100:
        return -100
    else:
        return senti_score


def sentiment_graph_generator(value):
    senti_value = []
    senti_labels = []
    senti_colors = []
    senti_explode = []

    if value > 0:
        senti_value.append(abs(value))
        senti_value.append(100 - abs(value))
        senti_labels.append("Pozitif Değer")
        senti_labels.append(None)
        senti_colors.append("#568E44")
        senti_colors.append("#c0c0c0")
        senti_explode.append(0.1)
        senti_explode.append(0)

    elif value < 0:
        senti_value.append(100 - abs(value))
        senti_value.append(abs(value))
        senti_labels.append(None)
        senti_labels.append("Negatif Değer")
        senti_colors.append("#c0c0c0")
        senti_colors.append("#8E4444")
        senti_explode.append(0)
        senti_explode.append(0.1)

    main_values = np.array(senti_value)
    plt.pie(
        main_values,
        startangle=180,
        shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.9},
        labels=senti_labels,
        colors=senti_colors,
        explode=senti_explode,
        autopct="%1.1f%%",
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64
