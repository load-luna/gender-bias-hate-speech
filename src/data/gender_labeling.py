import string

#create Gender Term DataFrame
gender_lexicon = {
    "male": {
       
            "he", "him", "his", "himself",
            
            "father","fathers", "dad","dads", "daddy","daddies", "son","sons", "brother","brothers", "uncle","uncles",
               
            "grandfather","grandfathers", "grandpa","grandpas", "husband","husbands", "boyfriend","boyfriends",
            "nephew", "stepfather", "stepdad", "halfbrother",
       
            "man", "men", "male","males", "boy", "boys", "gentleman", "gentlemen",
            "guy", "guys", "dude", "king", "prince", "sir",
            "actor", "waiter", "host", "businessman", "chairman",
            "policeman", "fireman", "salesman", "spokesman",
            "congressman", "statesman",
       
            "masculine", "manly", "virile", "macho", "fatherly",
            "boyish", "lad", "patriarch",
                 "gentlemanly","mr"
    },

    "female": {
            "she", "her", "hers", "herself",
                "babe",
            "mother","mothers", "mom","moms", "mommy","mommies", "daughter","daughters", "sister","sisters", "aunt","aunts",
            "grandmother","grandmothers", "grandma","grandmas", "wife","wives", "girlfriend","girlfriends",
            "niece", "stepmother", "stepmom", "halfsister",

            "woman", "women", "female","females", "girl", "girls", "lady", "ladies",
            "queen", "princess", "madam", "maam", "actress",
            "waitress", "hostess", "businesswoman", "chairwoman",
            "policewoman", "saleswoman", "spokeswoman",
            "congresswoman","firewoman",
            "lass", "ms", "madam",
            "ladylike",
        

            "feminine", "womanly", "girly", "ladylike",
            "maternal", "motherly", "matriarch", "diva"
        
    },

    "neutral": {
  
            "they", "them", "their", "theirs", "themselves",

            "parent", "parents", "child", "children", "kid", "kids",
            "sibling", "siblings", "spouse", "partner",
            "guardian", "relative",

            "person", "people", "individual", "human", "adult", "teen",
            "worker", "employee", "staff", "member", "leader",
            "manager", "chairperson", "police officer",
            "firefighter", "salesperson", "representative",
            "official", "citizen",

            "neutral", "androgynous", "nonbinary", "genderless",
            "inclusive"
        
    }
}

#swap gender in sentence
def swap_gender(sentence):
    # Define a mapping for swapping gender-specific words
    swap_dict = {
        # Male to female swaps
        'he': 'she',
        'him': 'her', 
        'his': 'her', 
        'himself': 'herself',
        "dude": "babe",
        "lad":"lass",
         #"mister": "miss", 
          "gentlemanly":"ladylike",
         "mr": "ms",
        'man': 'woman', 
        'men': 'women', 
        'male': 'female',
        'males': 'females',
        'boy': 'girl', 
        'boys': 'girls',
        'guy': 'girl', 
        'guys': 'girls', 
        'father': 'mother',
        'fathers': 'mothers', 
        'dad': 'mom',
        'dads': 'moms',
        "daddy": "mommy",
        "daddies": "mommies",
        'son': 'daughter',
        'sons': 'daughters', 
        'brother': 'sister', 
        'brothers': 'sisters', 
        'uncle': 'aunt',
        'uncles': 'aunts',
        'grandfather': 'grandmother',
        'grandfathers': 'grandmothers',  
        'grandpa': 'grandma', 
        'grandpas': 'grandmas',
        'husband': 'wife',
        'husbands': 'wives',
        'boyfriend': 'girlfriend', 
        'boyfriends': 'girlfriends', 
        'nephew': 'niece', 
        'stepfather': 'stepmother',
        'stepfathers': 'stepmothers',
        'stepdad': 'stepmom', 
        'stepdads': 'stepmoms', 
        "stepbrother": "stepsister",
        "stepbrothers": "stepsisters",
        'halfbrother': 'halfsister', 
        'halfbrothers': 'halfsisters', 
        'gentleman': 'lady',
        'gentlemen': 'ladies',
        'king': 'queen', 
        'kings': 'queens',
        'prince': 'princess', 
        'princes': 'princesses', 
        'sir': 'madam', 
        'actor': 'actress',
        'actors': 'actresses',
        'waiter': 'waitress', 
        'waiters': 'waitresses', 
        'host': 'hostess', 
        'hosts': 'hostesses', 
        'businessman': 'businesswoman',
        'businessmen': 'businesswomen',
        'chairman': 'chairwoman', 
        'chairmen': 'chairwomen', 
        'policeman': 'policewoman', 
        'policemen': 'policewomen', 
        'fireman': 'firewoman',
        'firemen': 'firewomen',
        'salesman': 'saleswoman', 
        'salesmen': 'saleswomen', 
        'spokesman': 'spokeswoman', 
        'spokesmen': 'spokeswomen', 
        'congressman': 'congresswoman',
        'congressmen': 'congresswomen',
        'statesman': 'stateswoman', 
        'statesmen': 'stateswomen', 
        'masculine': 'feminine', 
        'manly': 'womanly',
        'virile': 'feminine', 
        'macho': 'feminine', 
        'boyish': 'girly', 
        'patriarch': 'matriarch',


        # Female to male swaps
        'she': 'he', 
        'her': 'him', 
        'hers': 'his', 
        'herself': 'himself',
        "lass": "lad",
        #"miss": "mister",
        "diva": "drama king",
        "ladylike": "gentlemanly",
        "ms": "mr",
        "babe": "dude",
        'woman': 'man', 
        'women': 'men', 
        'female': 'male',
        'females': 'males',
        'girl': 'boy', 
        'girls': 'boys',
        'mother': 'father',
        'mothers': 'fathers',
        'mom': 'dad',
        'moms':'dads',
        "mommy": "daddy",
        "mommies": "daddies",
        'daughter': 'son',
        'daughters': 'sons',
        'sister': 'brother', 
        'sisters': 'brothers',
        'aunt': 'uncle',
        'aunts':'uncles',
        'grandmother':'grandfather',
        'grandmothers':'grandfathers', 
        'grandma': 'grandpa', 
        'grandmas':'grandpas',
        'wife': 'husband',
        'wives': 'husbands',
        'girlfriend':'boyfriend',
        'girlfriends': 'boyfriends', 
        'niece': 'nephew',
        "setepsister": "stepbrother",
        "stepsisters": "stepbrothers",
        'stepmother': 'stepfather',
        'stepmothers':'stepfathers',
        'stepmom': 'stepdad',
        'stepmoms': 'stepdads',
        'halfsister': 'halfbrother',
        'halfsisters': 'halfbrothers', 
        'lady': 'gentleman', 
        'ladies': 'gentlemen', 
        'queen': 'king',
        'queens': 'kings', 
        'princess': 'prince',
        'princesses': 'princes', 
        'madam': 'sir', 
        'maam': 'sir', 
        'actress': 'actor', 
        'actresses': 'actors', 
        'waitress': 'waiter',
        'waitresses': 'waiters',
        'hostess': 'host', 
        'hostesses': 'hosts', 
        'businesswoman': 'businessman', 
        'businesswomen': 'businessmen',
        'chairwoman': 'chairman',
        'chairwomen': 'chairmen',
        'policewoman': 'policeman', 
        'policewomen': 'policemen', 
        'firewoman': 'fireman',
        'firewomen': 'firemen',
        'saleswoman': 'salesman', 
        'saleswomen': 'salesmen',
        'spokeswoman': 'spokesman',
        'spokeswomen': 'spokesmen',
        'stateswoman': 'statesman', 
        'stateswomen': 'statesmen', 
        'congresswoman': 'congressman', 
        'congresswomen': 'congressmen',
        'feminine': 'masculine', 
        'womanly': 'manly',
        'girly': 'boyish', 
        'maternal': 'paternal', 
        'motherly': 'fatherly', 
        'matriarch': 'patriarch'
    }
    
    words = sentence.split()
    swapped_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in swap_dict:
            swapped_word = swap_dict[lower_word]
            swapped_words.append(swapped_word)
        else:
            swapped_words.append(word)
    return ' '.join(swapped_words)

def count_gender_terms(text):
    text = text.lower()
    counts = {"male": 0, "female": 0}
    for word in text.split():
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

#outdated
#needs Dataframe with column ["text"]
def gender_distribution(df):
    male_counts = 0
    female_counts = 0
    none_counts = 0
    equal_counts = 0
    label_avg = 0


    #found = False
    for index, row in df.iterrows():
        text = row["text"]
        #removing punctuation and making lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        swap_text = swap_gender(text)

        label = label_gender(text)
        swap_label = label_gender(swap_text)
        if label == 0:
            male_counts += 1
            
            if swap_label != 1:# and found == False:
                print("Mismatch in swap label for text:", text)
                print("swaped Text:", swap_text)
                #found = True
        elif label == 1:
            female_counts += 1
            if swap_label != 0:
                print("Mismatch in swap label for text:", text)
                print("swaped Text:", swap_text)
        elif label == 2:
            equal_counts += 1
            if swap_label != 2:
                print("Mismatch in swap label for text:", text)
                print("swaped Text:", swap_text)
        else:
            if swap_label != 3:
                print("Mismatch in swap label for text:", text)
                print("swaped Text:", swap_text)
            none_counts += 1
            

    #bitch
    #pussy
    #fiancé 
    #cunts
    #hoes
    avarage_male = male_counts*100 / len(df)
    avarage_female = female_counts*100 / len(df)
    avarage_none = none_counts*100 / len(df)
    avarage_equal = equal_counts*100 / len(df)

    print(f"ratio of tweets with male terms: {avarage_male}")
    print(f"ratio of tweets with female terms: {avarage_female}")
    print(f"ratio of tweets with equal gender label: {avarage_equal}")
    print(f"ratio of tweets with no gender terms: {avarage_none}")

    return {
            "Male" : avarage_male,
            "Female" : avarage_female,
            "Equal" : avarage_equal,
            "None" : avarage_none,
           }

#old
#count_label_distribution(df.to_dict(orient='records'))
def old_count_label_distribution(dataset):
    gender_types = 4
    predict_types = 2
    arr = [[0 for i in range(predict_types)] for j in range(gender_types)]
    
    for f in dataset:
        label = f['label']
        gender = f['gender_label']
        
        arr[gender][label] += 1
    #for i in range(gender_types):
       # for j in range(predict_types):
       #     arr[i][j] /= len(dataset)
       #     arr[i][j] *= 100
    print("male:", arr[0])
    print("female:", arr[1])
    print("equal:", arr[2])
    print("none:", arr[3])

    return arr

def count_label_distribution(dataset,config):
    gender_names = {
        0: "male",
        1: "female",
        2: "equal",
        3: "none",
    }

    
    label_names = config["dataset"]["label_mapping"]["names"]
    
    num_gender_labels = len(gender_names)
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

    for gender_id, gender_name in gender_names.items():
        print(
            f"{gender_name:10}",
            counts[gender_id]
        )

    return counts

#adds gender label columne to dataset
def add_gender_label(example):
    example["gender_label"] = label_gender(example["text"])
    return example
