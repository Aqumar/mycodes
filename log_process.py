import ast
import numpy as np
import pandas as pd
import shlex
import json
import re

count = 0
with open("log_all.out") as f_old, open("log_queries.out", "w") as f_new:
    for line in f_old:
        if 'payload=======:' in line:
            if count > 0:
                f_new.write("End\n")
            f_new.write("start\n")
            count = count + 1
            
        f_new.write(line)
        
    f_new.write("End\n")



with open('log_queries.out') as fp:
    lst = []
    for result in re.findall('start(.*?)End', fp.read(), re.S):
        lst.append(result)


count = 0
with open('result_all.json') as f, open("results_json.json", "w") as f_json:    
    for line in f:
        if count % 2 != 0:
            f_json.write(json.dumps(line))
            f_json.write("\n")
        count = count + 1

data = []
with open('results_json.json') as f:    
    for line in f:        
        data.append(json.loads(line))

all_queries = pd.read_csv('final_queries.csv')
data = pd.DataFrame(data,columns=['Response'])
query_response = pd.concat([all_queries,data],axis=1)
query_response = query_response.set_index('KEY')


df = pd.DataFrame(columns=['input_query','response','best_match','best_score',
                                        'second_best_match','second_best_score','exact_match'])

for i in range(len(lst)):
    temp = lst[i]
    if 'payload=======:' not in temp:
        continue

    if ('Fuzzy Query ===>' in temp) and ('Predictions ===>' in temp):
        t = (re.findall('payload=======:(.*?)Fuzzy', temp, re.S))
        input_query = ast.literal_eval(''.join(map(str, t)))['input']
        
        
        fuzzy_query = (re.findall('Fuzzy Query ===> (.*?)Sorted Prediction', temp, re.S))
        if len(fuzzy_query) == 2:
            fuzzy_query = fuzzy_query[0]
        fuzzy_query = ''.join(map(str, fuzzy_query))
        fuzzy_query = ast.literal_eval(fuzzy_query)
        

        predictions = (re.findall('Predictions ===> (.*?)Fuzzy Query', temp, re.S))
        if len(predictions) == 2:
            predictions = predictions[0]
        predictions = ''.join(map(str, predictions))
        predictions = predictions.replace(']','')
        predictions = predictions.replace('[','')
        predictions = ','.join(shlex.split(predictions))
        predictions = ast.literal_eval(predictions)
        
        if type(predictions) == float:
            best_score = predictions
            best_match = fuzzy_query[0]
            second_best_score = best_score
            second_best_match = best_match
            
            response_json = query_response.loc[input_query]['Response']
            response = ast.literal_eval(ast.literal_eval(response_json))['response']['response']
            
            df = df.append({'input_query': input_query, 'response':response, 'best_match':best_match, 'best_score':best_score,
                        'second_best_match':second_best_match, 'second_best_score':second_best_score, 
                        'exact_match': False}, ignore_index=True)
            continue

        predictions = list(predictions)
        
        

        best_score = min(predictions)
        best_match = fuzzy_query[predictions.index(best_score)]
        if len(set(predictions)) == 1:
            second_best_score = best_score
            second_best_match = best_match
        else:
            second_best_score = np.amin(np.array(predictions)[predictions != np.amin(predictions)])
            second_best_match = fuzzy_query[predictions.index(second_best_score)]
            
        response_json = query_response.loc[input_query]['Response']
        response = ast.literal_eval(ast.literal_eval(response_json))['response']['response']
        
        df = df.append({'input_query': input_query, 'response':response, 'best_match':best_match, 'best_score':best_score,
                        'second_best_match':second_best_match, 'second_best_score':second_best_score, 
                        'exact_match': False}, ignore_index=True)

set_all_queries = set(all_queries['KEY'])
set_df_queries = set(df['input_query'])
exact_match_queries = set_all_queries.difference(set_df_queries)
exact_match_queries = list(exact_match_queries)

df_exact_match = pd.DataFrame(columns=['input_query','response','best_match','best_score',
                                        'second_best_match','second_best_score','exact_match'])
for i in range(len(exact_match_queries)):
    try:
        input_query = exact_match_queries[i]
    
        response_json = query_response.loc[input_query]['Response']
        response = ast.literal_eval(ast.literal_eval(response_json))['response']['response']
    
        df_exact_match = df_exact_match.append({'input_query': input_query, 'response':response, 
                                                'best_match':'NA', 'best_score':'NA',
                                                'second_best_match':'NA', 'second_best_score':'NA', 
                                                'exact_match': True}, ignore_index=True)
    except ValueError:
        print("Malformed Response")

    
    
df_all = pd.concat([df,df_exact_match],axis=0)

df_all.to_csv('processed_log_file.csv',sep=',')

