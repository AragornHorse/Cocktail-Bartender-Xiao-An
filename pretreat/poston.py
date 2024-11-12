import pandas as pd
import numpy as np
import re
import json
import filter
import clusters

its = set()

pth = r"D:\Users\DELL\Desktop\datasets\cocktail\ori\mr-boston-flattened.csv"
tgt = r"D:\Users\DELL\Desktop\datasets\cocktail\jsons\mr-boston-flattened.json"
npy = r"D:\Users\DELL\Desktop\datasets\cocktail\npy\\"


def csv_to_json():
    df = pd.read_csv(pth)

    datas = []

    for index, row in df.iterrows():
        rec = {
            'name': row['name'].strip(),
            'contents': {}
        }
        for i in range(1, 7):
            name = row[f'ingredient-{i}']
            num = row[f"measurement-{i}"]

            # empty item
            if str(name) == 'nan':
                continue

            # don't know number
            if str(num) == 'nan':
                rec['contents'][name.strip()] = None
                continue

            # find number
            nu = re.findall(r"(\d+/{0,1}\d*)", num)   # oz or splash

            # have number
            if len(nu) > 0:

                # 1 oz
                if len(nu) == 1:
                    nu = float(eval(nu[0]))

                # 1 1/2 oz
                elif len(nu) == 2:
                    nu = float(eval(nu[0])) + float(eval(nu[1]))

                # 1 or 2 oz
                if 'or' in num:
                    nu = nu / 2

                # oz or ml
                if 'oz' in num:
                    pass
                elif 'dash' in num:
                    nu = nu / 36
                elif 'tsp' in num:
                    nu = nu * 0.17
                elif 'drop' in num:
                    nu = nu * 0.001
                elif 'splash' in num:
                    nu = nu * 0.845
                elif 'ml' in num:
                    nu = re.findall(r"(\d+/{0,1}\d*)", num)
                    nu = float(eval(nu[0])) * float(eval(nu[1])) / 30
                else:
                    if 'Orange half-wheel' in name:
                        name = 'Orange'
                        nu = 0.5
                    elif 'olive' in str(name):
                        name = 'Olive'
                        nu = 0.5
                    elif 'mint' in name.lower():
                        name = 'Mint'
                        if nu < 7:
                            nu = 0.5
                        else:
                            nu = 1.
                    elif 'egg' in name.lower():
                        if 'white' in name.lower():
                            nu = 1.5
                        else:
                            nu = 2.
                        name = 'Egg'
                    elif 'wedge' in name.lower():
                        nu = nu * 0.1
                        name = 'Lime'
                    elif 'Juice of Lemon or Lime' in name:
                        name = 'Lemon Juice'
                        nu = 1.
                    elif 'ketchup' in name:
                        name = 'Lemon Juice'
                        nu = 0.5
                    elif 'Sweet pickle' in name:
                        name = 'Pickle'
                        nu = 1.
                    elif 'red wine' in name:
                        name = 'Red Wine'
                        nu = 1500 / 30
                    elif 'Cucumber slices' in name:
                        name = 'Cucumber'
                        nu = 1.
                    elif 'lump of sugar' in name:
                        name = 'Sugar'
                        nu = 4.5 / 30
                    elif 'cloves' in name:
                        name = 'Cloves'
                        nu = 0.1
                    elif 'fresh ginger' in name:
                        name = 'Ginger'
                        nu = 0.5
                    elif 'orange peel' in name:
                        name = 'Orange Peel'
                        nu = 0.3
                    elif 'pineapple' in name:
                        name = 'Pineapple'
                        nu = 1.5
                    else:
                        rec['contents']['Strawberry'] = 0.5
                        rec['contents']['Peach'] = 0.3
                        rec['contents']['Cherry'] = 0.2
                        continue

                rec['contents'][name.strip()] = nu

            # no number
            else:
                others = name.split(',')
                others = [o.strip() for o in others]
                for o in others:
                    rec['contents'][o] = None

        # filter name
        rec_ = {'name': rec['name'], 'contents': {}}
        for k, v in rec['contents'].items():
            k_ = filter.filter(k)
            # if k_ is None:
            #     print(k)
            rec_['contents'][k_] = rec['contents'][k]
            # if k_ is None:
            # print(k_)

        datas.append(rec_)

    sets = set()
    for rec in datas:
        for ct in rec['contents'].keys():
            sets.add(ct)


    with open(tgt, 'w', encoding='utf-8') as f:
        json.dump(datas, f)

csv_to_json()

with open(tgt, 'r', encoding='utf-8') as f:
    datas = json.load(f)

# fill NaN
for rec in datas:
    for ct, num in rec['contents'].items():
        if num is None:
            if 'soda' in ct.lower() or 'beer' in ct.lower():
                nums = [n for k, n in rec['contents'].items() if n is not None]
                rec['contents'][ct] = np.clip(max(nums), 0.25, 2.)
            elif 'peel' in ct.lower():
                rec['contents'][ct] = 1.
            else:
                rec['contents'][ct] = 1.

# get bad cts
id_to_ct = set()
for rec in datas:
    for ct in rec['contents'].keys():
        id_to_ct.add(ct)
id_to_ct = list(id_to_ct)
ct_to_id = {k: v for v, k in enumerate(id_to_ct)}

id_to_name = [rec['name'] for rec in datas]
name_to_id = {k: v for v, k in enumerate(id_to_name)}

data_array = np.zeros([len(datas), len(id_to_ct)])

for rec in datas:
    for ct, num in rec['contents'].items():
        data_array[name_to_id[rec['name']], ct_to_id[ct]] = num

appear_time = np.sum((data_array > 0).astype(int), axis=0)
bad_ct = np.nonzero((appear_time < 2).astype(int))[0]
bad_ct = [id_to_ct[idx] for idx in bad_ct]


# remove cocktail having bad ct
datas_ = []

for i, rec in enumerate(datas):
    for ct in rec['contents'].keys():
        if ct in bad_ct:
            break
    else:
        datas_.append(rec)

datas = datas_

# get mat
id_to_ct = set()
for rec in datas:
    for ct in rec['contents'].keys():
        id_to_ct.add(ct)
id_to_ct = list(id_to_ct)
ct_to_id = {k: v for v, k in enumerate(id_to_ct)}

id_to_name = [rec['name'] for rec in datas]
name_to_id = {k: v for v, k in enumerate(id_to_name)}
data_array = np.zeros([len(datas), len(id_to_ct)])

for rec in datas:
    for ct, num in rec['contents'].items():
        data_array[name_to_id[rec['name']], ct_to_id[ct]] = num

data_array = data_array / np.sum(data_array, axis=1, keepdims=True)

print(data_array.shape)

np.save(npy + "\mat.npy", data_array)
with open(npy + "\\name2idx.json", 'w', encoding='utf-8') as f:
    json.dump(name_to_id, f)
with open(npy + "\\ct2idx.json", 'w', encoding='utf-8') as f:
    json.dump(ct_to_id, f)




