import json
import os

if __name__ == '__main__':
    print("hello")
    ontology = open('/home/fzus/lyh/DiCoS/data/mwz2.1/ontology.json', 'r', encoding='utf-8')
    schema = open('/home/fzus/lyh/DiCoS/data/mwz2.1/schema_back.json', 'r', encoding='utf-8')

    ontology = json.load(ontology)
    schema = json.load(schema)
    new_ontology = {}
    for s in ontology.keys():
        if 'price' in s or 'leave' in s or 'arrive' in s:
                new_key = s.replace(' ', '')
        new_ontology[new_key] = ontology[s]

    for i, k in enumerate(schema):
        d = k['service_name']

        slots = k['slots']
        for s in slots:
            if 'price' in s['name'] or 'leave' in s['name'] or 'arrive' in s['name']:
                s['name'] = s['name'].replace(' ', '')
            if s['name'] in new_ontology.keys():
                s['possible_values'] = new_ontology[s['name']]
    
    json_str = json.dumps(schema, indent=4)
    with open('schema_.json', 'w', encoding='utf-8') as json_file:
         json_file.write(json_str)
            



    