import numpy as np
import openai, json, sys

np.random.seed(42)

# set up key
openai.api_key = open('gpt3apikey.txt').readlines()[0]

def process_example_lines(example_lines, total_data_to_keep=3):
    category = example_lines[0].replace("\n","").replace("#","")
    example_data_lines = list()
    example_request = ""
    example_answer = ""
    example_entities = list()
    for j, eline in enumerate(example_lines[1:]):
        if eline.startswith("0 "):
            # fix-ups for entities that have "'" or "," within them
            processed_eline = eline[2:].replace('"',"'").replace("'",'').replace('\n','').replace('_,_','_')
            example_data_lines.append(processed_eline)
        else:
            splitted_eline = eline.split('\t')
            example_request = splitted_eline[0][2:]
            example_answer = splitted_eline[1]
            eents = splitted_eline[2]
            # do some fix-ups on eents if they have "'" or "," within them
            example_entities = eents.replace('[','').replace(']','').replace('"',"'").replace(' ','').replace("'",'').replace('\n','').replace('_,_','_').split(',')
    
    example = {
        'data': example_data_lines,
        'category': category,
        'request': example_request,
        'answer': example_answer,
        'entities': example_entities
    }

    # trim off data
    example['data'] = trim_off_data(data=example['data'], category=example['category'], relevant_entities=example['entities'], keep=total_data_to_keep)

    # generate prompt
    prompt = "DATA:\n"
    prompt += "\n".join(example["data"])
    prompt += "\nQUESTION:\n"
    prompt += example["request"]
    prompt += "\nANSWER:\n"
    prompt_w_answer = prompt+example["answer"]+"\n###\n"

    example['prompt'] = prompt
    example['prompt_w_answer'] = prompt_w_answer

    return example

def trim_off_data(data, category, relevant_entities, keep=3):
    # separate data into chunks 
    data_chunks = list()
    prev_data_item_last_line = 0

    if category == 'weather': 
        data = data[1:] # exclude "today monday" suffix

    weather_or_schedule_prev_name = None
    for i, line in enumerate(data):
        condition = False
        if category in ['hotel','restaurant','attraction','navigate']: 
            if len(line.split(' ')) > 3:
                condition=True
        elif category in ['weather','schedule']: 
            new_name = line.split(' ')[0]
            if weather_or_schedule_prev_name is not None and weather_or_schedule_prev_name != new_name:
                condition = True
            weather_or_schedule_prev_name = new_name
            
        if not condition and i != len(data)-1:
            continue

        next_idx_to_exclude = i if i != len(data)-1 else i+1
        # record what processed so far
        data_chunk = data[prev_data_item_last_line:next_idx_to_exclude] 

        prev_data_item_last_line = i
        if len(data_chunk) == 0:
            continue
        data_chunks.append(data_chunk)

    # remove data chunks
    while keep != 'all' and len(data_chunks) > keep:
        random_idx = np.random.randint(len(data_chunks))
        # understand if this data chunk must be kept because it is the entity at play
        must_be_kept = check_data_chunk_must_be_kept(data_chunks[random_idx], category, relevant_entities)
        if not must_be_kept:
            data_chunks.pop(random_idx)

    reassambled_data = list()
    for data_chunk in data_chunks:
        reassambled_data += data_chunk

    return reassambled_data
        

def check_data_chunk_must_be_kept(data_chunk, category, relevant_entities):
    if category == 'hotel': 
        price = relevant_entities[0]
        name = relevant_entities[1]
        if price in data_chunk[0] and name in data_chunk[0]:
            #print(data_chunk[0], price, name)
            return True 
    elif category == 'attraction': 
        attraction_name = relevant_entities[0]
        street = relevant_entities[1]
        area = relevant_entities[2]
        if attraction_name in data_chunk[0] and street in data_chunk[0] and area in data_chunk[0]:
            return True
    elif category == 'restaurant': 
        price = relevant_entities[0]
        location_type = relevant_entities[1]
        cousine = relevant_entities[2]
        if price in data_chunk[0] and location_type in data_chunk[0] and cousine in data_chunk[0]:
            return True
    elif category == 'weather': 
        place = relevant_entities[0]
        if place in data_chunk[0]:
            return True
    elif category == 'navigate': 
        place_name = relevant_entities[1]
        if place_name in data_chunk[0]:
            return True
    elif category == 'schedule': 
        event = relevant_entities[1]
        if event in data_chunk[0]:
            return True
    
    return False


def load_data_set(path):
    lines = open(test_path).readlines()
    examples = list()
    prev_example_last_line = 0
    for i, line in enumerate(lines):
        if line != '\n' and i != len(lines)-1:
            continue
        next_idx_to_exclude = i if i != len(lines)-1 else i+1
        # record what processed so far
        example_lines = lines[prev_example_last_line:next_idx_to_exclude]
        example = process_example_lines(example_lines, 3)
        prev_example_last_line = i+1
        examples.append(example)
    return examples

    
name = sys.argv[1]
test_path = 'data/'+('KVR' if name=='kvr' else 'MULTIWOZ2.1')+'/our_test.txt'
train_path = 'data/'+('KVR' if name=='kvr' else 'MULTIWOZ2.1')+'/our_train.txt'

test_set = load_data_set(test_path)
train_set = load_data_set(train_path)

# some preparations to compute scores
categories = np.unique([x['category'] for x in train_set])
all_entities = {}
skipped_samples = {}
fp = {}
fn = {}
tp = {}
for cat in categories:
    tp[cat] = 0
    fp[cat] = 0
    fn[cat] = 0
    all_entities[cat] = list()
    for example in test_set + train_set:
        if example['category'] == cat:
            all_entities[cat] += example['entities']
    skipped_samples[cat] = 0

np.random.shuffle(test_set)
# for each test example, generate prompt for GPT-3, and get results!
for test_example in test_set:
    # find a relevant training example that is not exactly the same
    relevant_training_example = test_example
    while(relevant_training_example['request'] == test_example['request']):
        relevant_training_examples = [x for x in train_set if x['category'] == test_example['category']]
        np.random.shuffle(relevant_training_examples)
        relevant_training_example = relevant_training_examples[0]

    irrelavant_training_examples = [x for x in train_set if x['category'] != test_example['category']]
    np.random.shuffle(irrelavant_training_examples)
    irrelavant_training_example = irrelavant_training_examples[0]


    # take one relevant and one irrelevant
    rel_n_irr_train_ex = [relevant_training_example, irrelavant_training_example]
    np.random.shuffle(rel_n_irr_train_ex)

    # built tranining prompt
    training_prompt = "".join([x['prompt_w_answer'] for x in rel_n_irr_train_ex])
    question_prompt = test_example['prompt']
    total_prompt = training_prompt + question_prompt
    
    # submit prompt to GPT-3
    if True:
        try:
            response = openai.Completion.create(engine="davinci", 
                prompt=total_prompt, 
                temperature=0.0, top_p=1.0,
                stop=['\n'],
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=50)
            answer = response['choices'][0]['text']
        except:
            print("Warning, skipping sample! Try reducing the number of data chunks in the prompts!")
            skipped_samples[test_example['category']] += 1
            continue
    else:
        answer = 'menlo_park is 1_miles away'

    print(question_prompt[-51:]+answer)

    # compute false positive, false negative, etc.
    entities_of_the_category = all_entities[test_example['category']]
    entities_of_other_categories = list()
    for cat in all_entities:
        if cat == test_example['category']:
            continue
        entities_of_other_categories += all_entities[cat] 

    retrieved_entities = [x for x in answer.split(' ') if x in entities_of_the_category] + [x for x in answer.split(' ') if x in entities_of_other_categories]

    for entity in test_example['entities']:
        if entity in retrieved_entities:
            tp[test_example['category']] += 1
        else:
            fn[test_example['category']] += 1
    
    for entity in set(retrieved_entities):
        if entity not in test_example['entities']:
            fp[test_example['category']] +=1


# print results
total_tp = np.sum(list(tp.values()))
total_fp = np.sum(list(fp.values()))
total_fn = np.sum(list(fn.values()))

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) != 0 else 0 
recall  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) != 0 else 0 
f1 = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0
print("F1", np.round(f1*100,1))
print("Prec", np.round(precision*100,1))
print("Rec", np.round(recall*100,1))
for cat in categories:
    print(cat)
    precision = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) != 0 else 0 
    recall  = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) != 0 else 0 
    f1 = 2*precision*recall / (precision + recall) if (precision + recall) != 0 else 0
    print("\tF1", np.round(f1*100,1))
    print("\tPrec", np.round(precision*100,1))
    print("\tRec", np.round(recall*100,1))    

print(skipped_samples)