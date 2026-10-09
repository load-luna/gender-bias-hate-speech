import re
import codecs

#before tokenization
#dataset.map(deep_clean_tweet)
def deep_clean_tweet(example):

    text = example["text"]  
    if not isinstance(text, str):
        return example["text"]

    text = text.lower()  # Convert to lowercase for uniformity

    # Step 1: Remove URLs immediately to avoid confusing the decoder
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'http?://\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    #html tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)


    # Step 2: Recursive Unescape
    # This loop keeps decoding until no more escape sequences are found
    last_text = None
    while text != last_text:
        last_text = text
        try:
            # Fix double-slashes and literal hex/unicode escapes
            # Using codecs.decode is often more robust for literal strings
            text = codecs.decode(text, 'unicode_escape')
            
            # If it looks like raw UTF-8 bytes (like \xe7), fix the encoding shift
            if '\\x' in text or any(ord(c) > 128 for c in text):
                text = text.encode('latin-1').decode('utf-8')
        except:
            break

    # Step 3: Handle specific messy artifacts left behind
    replacements = {
        "\\\'": "'",    # Fix escaped apostrophes
        "\\\"": '"',    # Fix escaped quotes
        "\xa0": " ",    # Fix non-breaking spaces
        "\\'": "'", 
        "\\ ": " ",
        "\\/": "/",
        "\\_": "_",
        "\\p": "p",
        "\\(": "(",
        "*" : "",
        r"{}".format("\\") : "",
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    ##NOT WORKING! :/
    #removing all punctiuation
    #text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Final cleanup of whitespace and newlines

    text = ' '.join(text.split())
    example["text"] = text.strip()
    return example


