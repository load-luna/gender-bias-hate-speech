import string

import re

#create Gender Term DataFrame
# Format: (male, female)
gender_pairs = [
    # Pronomen mit eindeutiger Zuordnung
    ("he", "she"),
    ("himself", "herself"),

    # Anreden und Personenbezeichnungen
    ("mr", "ms"),
    ("man", "woman"),
    ("men", "women"),
    ("male", "female"),
    ("males", "females"),
    ("boy", "girl"),
    ("boys", "girls"),
    ("gentleman", "lady"),
    ("gentlemen", "ladies"),
    ("lad", "lass"),
    ("dude", "babe"),
    ("sir", "madam"),
    ("guy", "gal"),
    ("guys", "gals"),

    # Familie und Beziehungen
    ("father", "mother"),
    ("fathers", "mothers"),
    ("dad", "mom"),
    ("dads", "moms"),
    ("daddy", "mommy"),
    ("daddies", "mommies"),
    ("son", "daughter"),
    ("sons", "daughters"),
    ("brother", "sister"),
    ("brothers", "sisters"),
    ("uncle", "aunt"),
    ("uncles", "aunts"),
    ("grandfather", "grandmother"),
    ("grandfathers", "grandmothers"),
    ("grandpa", "grandma"),
    ("grandpas", "grandmas"),
    ("husband", "wife"),
    ("husbands", "wives"),
    ("boyfriend", "girlfriend"),
    ("boyfriends", "girlfriends"),
    ("nephew", "niece"),
    ("stepfather", "stepmother"),
    ("stepfathers", "stepmothers"),
    ("stepdad", "stepmom"),
    ("stepdads", "stepmoms"),
    ("stepbrother", "stepsister"),
    ("stepbrothers", "stepsisters"),
    ("halfbrother", "halfsister"),
    ("halfbrothers", "halfsisters"),

    # Titel
    ("king", "queen"),
    ("kings", "queens"),
    ("prince", "princess"),
    ("princes", "princesses"),

    # Berufe und Rollen
    ("actor", "actress"),
    ("actors", "actresses"),
    ("waiter", "waitress"),
    ("waiters", "waitresses"),
    ("host", "hostess"),
    ("hosts", "hostesses"),
    ("businessman", "businesswoman"),
    ("businessmen", "businesswomen"),
    ("chairman", "chairwoman"),
    ("chairmen", "chairwomen"),
    ("policeman", "policewoman"),
    ("policemen", "policewomen"),
    ("fireman", "firewoman"),
    ("firemen", "firewomen"),
    ("salesman", "saleswoman"),
    ("salesmen", "saleswomen"),
    ("spokesman", "spokeswoman"),
    ("spokesmen", "spokeswomen"),
    ("congressman", "congresswoman"),
    ("congressmen", "congresswomen"),
    ("statesman", "stateswoman"),
    ("statesmen", "stateswomen"),
    ("patriarch", "matriarch"),

    # Eigenschaften
    ("masculine", "feminine"),
    ("manly", "womanly"),
    ("boyish", "girly"),
    ("fatherly", "motherly"),
    ("paternal", "maternal"),
    ("gentlemanly", "ladylike"),
]





# Kollidieren mit bereits verwendeten Gegenstücken:
# maam → sir        (sir ist bereits mit madam gepaart)


# Wörter mit Genderbezug die nicht eindeutig getauscht werdenen / haben beleidigenden beigeschmack
#bitch
#pussy
#cunts
#hoes
#diva 

#lexicon of gender term labeling
gender_lexicon = {
    "male": {male for male, female in gender_pairs},
    "female": {female for male, female in gender_pairs},
}



#swap Dictionary for gender swapping
swap_dict = {}
for male, female in gender_pairs:
    if male == female or male in swap_dict or female in swap_dict:
        raise ValueError(f"Keine eindeutige Zuordnung: {male}, {female}")

    swap_dict[male] = female
    swap_dict[female] = male



# Benötigen eine grammatische Unterscheidung:
# him ↔ her     (Objekt)
# his ↔ her     (besitzanzeigend vor einem Nomen)
# his ↔ hers    (selbstständiges Possessivpronomen)
swap_dict.update({
    "him": "her",
    "her": "him",   # Unsicher: possessives "her" müsste zu "his" werden
    "his": "her",   # Unsicher: selbstständiges "his" müsste zu "hers" werden
    "hers": "his",
})

gender_lexicon["male"].update({"him", "his"})
gender_lexicon["female"].update({"her", "hers"})

grammar_uncertain_terms = {"her", "his"}

gender_labels = {
    0: "male",
    1: "female",
    2: "equal",
    3: "none",
}

def get_grammar_uncertain_terms(text):
    return sorted(set(get_words(text)) & grammar_uncertain_terms)

def get_word_matches(text):
    return text, list(re.finditer(r"[^\W_]+", text))


def get_words(text):
    """Return lowercase words using the shared word recognition rules."""
    _,matches = get_word_matches(text)
    return [match.group(0).lower() for match in matches]


def normalize_gender_pairs(pairs):
    """Validate and normalize directed (male_term, female_term) pairs."""
    normalized = []
    for pair in pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each held-out gender pair must contain two terms.")
        male, female = (str(term).lower() for term in pair)
        if not male or not female or swap_dict.get(male) != female or swap_dict.get(female) != male:
            raise ValueError(f"Terms do not form a reversible gender pair: {pair!r}")
        if (male, female) not in normalized:
            normalized.append((male, female))
    return normalized


def contains_gender_pair_term(text, pairs):
    """Whether text contains any exact token from the configured gender pairs."""
    terms = {term for pair in normalize_gender_pairs(pairs) for term in pair}
    return bool(terms.intersection(get_words(text)))


def gender_pair_side_presence(text, pair):
    """Return (male-side-present, female-side-present) for one exact word pair."""
    male, female = normalize_gender_pairs([pair])[0]
    words = set(get_words(text))
    return male in words, female in words


def _swap_with_dict(sentence, replacements):
    sentence, matches = get_word_matches(sentence)
    parts = []
    previous_end = 0
    for match in matches:
        word = match.group(0)
        replacement = replacements.get(word.lower())
        if replacement is None:
            replacement = word
        elif word.isupper():
            replacement = replacement.upper()
        elif word.istitle():
            replacement = replacement.capitalize()
        # Copy everything between words, including punctuation and underscores.
        parts.append(sentence[previous_end:match.start()])
        parts.append(replacement)
        previous_end = match.end()
    parts.append(sentence[previous_end:])
    return "".join(parts)


def swap_gender(sentence):
    return _swap_with_dict(sentence, swap_dict)


def swap_gender_for_pairs(sentence, pairs):
    """Swap only the terms in the selected pairs; leave all other terms intact."""
    replacements = {}
    for male, female in normalize_gender_pairs(pairs):
        replacements[male] = female
        replacements[female] = male
    return _swap_with_dict(sentence, replacements)


def count_gender_terms(text):
    counts = {"male": 0, "female": 0}
    for word in get_words(text):
        if word in gender_lexicon["male"]:
            counts["male"] += 1
        elif word in gender_lexicon["female"]:
            counts["female"] += 1
    return counts


def label_gender(text):
    counts = count_gender_terms(text)
    if counts["male"] > counts["female"]:
        return 0 #male
    elif counts["female"] > counts["male"]:
        return 1 #female
    elif counts["male"] == 0 and counts["female"] == 0:
        return 3 #no gender
    elif counts["male"] == counts["female"]:
        return 2 #equal
    else:
        raise ValueError("Unexpected case in label_gender function")


def count_label_distribution(dataset,config):

    label_names = config["dataset"]["label_mapping"]["names"]
    
    num_gender_labels = len(gender_labels)
    num_class_labels = len(label_names)

    counts = [
        [0 for _ in range(num_class_labels)]
        for _ in range(num_gender_labels)
    ]

    for example in dataset:
        label = example["label"]
        gender_label = example["gender_label"]

        counts[gender_label][label] += 1

    print(" " * 10, list(label_names.values()))

    for gender_id, gender_name in gender_labels.items():
        print(
            f"{gender_name:10}",
            counts[gender_id]
        )

    return counts

#adds gender label columne to dataset
def add_gender_label(example,*,output_column_name="gender_label", input_column_name="text"):
    example[output_column_name] = label_gender(example[input_column_name])
    return example

def swap_gender_terms_dataset(example,*, column_name="text"):
    original_text = example[column_name]
    uncertain_terms = get_grammar_uncertain_terms(original_text)

    example["swap_grammar_uncertain"] = bool(uncertain_terms)
    example["swap_grammar_uncertain_terms"] = uncertain_terms

    swapped = swap_gender(example[column_name])
    example[column_name] = swapped
    return example
