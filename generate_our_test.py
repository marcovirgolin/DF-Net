import json
import numpy as np

# for reproducibility (test examples will be random)
np.random.seed(4200)

entities_kvr = json.load(open('data/KVR/kvret_entities.json'))
entities_multiwoz = json.load(open('data/MULTIWOZ2.1/global_entities.json'))

''' Creates alternative formulations based on the training examples '''
def process_training_set(file_path):

    examples = list()
    ll = open(file_path).readlines()

    prev_example_last_line = 0
    for i, line in enumerate(ll):
        if line != '\n' and i != len(ll)-1:
            continue
        next_idx_to_exclude = i if i != len(ll)-1 else i+1
        # record what processed so far
        example_lines = ll[prev_example_last_line:next_idx_to_exclude]
        prev_example_last_line = i+1

        example_type = example_lines[0].replace("\n","")
        alternatives = get_alternatives(example_lines)
        examples.append({'example_type:': example_type, 'example_lines': example_lines, 'alternatives': alternatives})

    return examples


def get_alternatives(example_lines):
    # find entities in use
    entities_in_use = [x.replace("[","").replace("]","").replace("'","").replace("\n","").strip() for x in example_lines[-1].split('\t')[-1].split(',')]
    # find type of example 
    example_type = example_lines[0].replace("\n","")
    # user utterance 
    utterances = example_lines[-1].split('\t')[0:2]
    # kb_lines
    kb_lines = example_lines[1:-2]

    if example_type == '#restaurant#':
        alternatives = get_alternatives_restaurant(kb_lines, entities_in_use, utterances)
    elif example_type == "#hotel#":
        alternatives = get_alternatives_hotel(kb_lines, entities_in_use, utterances)
    elif example_type == "#attraction#":
        alternatives = get_alternatives_attraction(kb_lines, entities_in_use, utterances)
    elif example_type == "#navigate#":
        alternatives = get_alternatives_navigate(kb_lines, entities_in_use, utterances)
    elif example_type == "#weather#":
        alternatives = get_alternatives_weather(kb_lines, entities_in_use, utterances)

    return alternatives

def get_alternatives_restaurant(orig_kb_lines, entities_in_use, utterances):
    kb_lines = [x.replace("\n","").split(' ') for x in orig_kb_lines]
    kb_lines = [x[1:] for x in kb_lines]
    kb_lines = [x for x in kb_lines if len(x) > 3]
    request = utterances[0]
    response = utterances[1]
    restaurant_name = [x for x in request.split(" ") if '_' in x][0]
    
    price = entities_in_use[0]
    location_type = entities_in_use[1]
    cousine = entities_in_use[2]
    
    alternatives = list()
    for kb_line in kb_lines:
        if kb_line[0] == restaurant_name:
            continue
        else:
            alt_name = kb_line[0]
            alt_location_type = kb_line[6]
            alt_cousine = kb_line[2]
            alt_price = kb_line[5]
        alt_entities = [alt_price, alt_location_type, alt_cousine]
        alt_request = request.replace(restaurant_name, alt_name)
        alt_response = response.replace(price, alt_price).replace(location_type, alt_location_type).replace(cousine, alt_cousine)

        alternative_line = "\t".join([alt_request, alt_response, str(alt_entities)])
        alternative_example = "#restaurant#\n"+"".join(orig_kb_lines)+alternative_line
        alternatives.append(alternative_example)

    return alternatives

def get_alternatives_hotel(orig_kb_lines, entities_in_use, utterances):
    kb_lines = [x.replace("\n","").split(' ') for x in orig_kb_lines]
    kb_lines = [x[1:] for x in kb_lines]
    kb_lines = [x for x in kb_lines if len(x) > 3]
    
    request = utterances[0]
    response = utterances[1]

    possible_prices = ['cheap','moderate','expensive']
    price = entities_in_use[0]
    hotel_name = entities_in_use[1]

    alternatives = list()
    for kb_line in kb_lines:
        if kb_line[0] == hotel_name:
            continue
        else:
            alt_name = kb_line[-1]
            alt_price = kb_line[4]
        alt_entities = [alt_price, alt_name]
        alt_request = request.replace(price, alt_price)
        alt_response = response.replace(price, alt_price).replace(hotel_name, alt_name)

        alternative_line = "\t".join([alt_request, alt_response, str(alt_entities)])

        # Fix the KB so that only 1 entry in has the desired price range 
        # (because DF-Net's code does not allow for multiple possible answers)
        new_kb_lines = [x for x in orig_kb_lines]
        for i, okb_line in enumerate(orig_kb_lines):
            if alt_name in okb_line:
                #nkb_line = okb_line.replace('expensive',alt_price).replace('moderate',alt_price).replace('cheap',alt_price) # fix this example
                #new_kb_lines[i] = nkb_line
                continue
            if alt_price in okb_line:
                # randomizing the new price only for the ID of the KB, else use the one used before
                if len(okb_line.split(' ')) > 4:
                    other_prices = [x for x in possible_prices if x != alt_price]
                    other_price = np.random.choice(other_prices)
                    prev_price_choice = other_price
                else:
                    other_price = prev_price_choice
                nkb_line = okb_line.replace(alt_price, other_price)
                new_kb_lines[i] = nkb_line

        alternative_example = "#hotel#\n"+"".join(new_kb_lines)+alternative_line
        alternatives.append(alternative_example)

    return alternatives


def get_alternatives_attraction(orig_kb_lines, entities_in_use, utterances):
    kb_lines = [x.replace("\n","").split(' ') for x in orig_kb_lines]
    kb_lines = [x[1:] for x in kb_lines]
    kb_lines = [x for x in kb_lines if len(x) > 3]
    
    request = utterances[0]
    response = utterances[1]

    attraction_name = entities_in_use[0]
    street = entities_in_use[1]
    area = entities_in_use[2]

    alternatives = list()
    for kb_line in kb_lines:
        if kb_line[-1] == attraction_name:
            continue
        else:
            alt_name = kb_line[-1]
            alt_street = kb_line[0]
            alt_area = kb_line[1]
        alt_entities = [alt_name, alt_street, alt_area]
        alt_request = request.replace(attraction_name, alt_name)
        alt_response = response.replace(attraction_name, alt_name).replace(street, alt_street).replace(area, alt_area)

        alternative_line = "\t".join([alt_request, alt_response, str(alt_entities)])
        alternative_example = "#attraction#\n"+"".join(orig_kb_lines)+alternative_line
        alternatives.append(alternative_example)

    return alternatives


def get_alternatives_navigate(kb_lines, entities_in_use):
    pass

#path = 'data/KVR/our_train.txt'
path = 'data/MULTIWOZ2.1/our_train.txt'
examples = process_training_set(path)
test_path = 'data/MULTIWOZ2.1/our_test.txt'

our_test = ""
all_alternatives = list()
for example in examples:
    all_alternatives += example['alternatives']
np.random.shuffle(all_alternatives)
for i, alternative in enumerate(all_alternatives):
    our_test += alternative
    if i < len(all_alternatives) - 1:
        our_test += "\n\n"
open(test_path, 'w').write(our_test)