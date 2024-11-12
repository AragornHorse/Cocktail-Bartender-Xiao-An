import json
import pandas as pd


#     0. 划分小类，永远一起出现的配对合并
#     1. 划分大类，类间可共存，类内不可共存
#         · 先预处理，去掉不存在的小类、合并显然相同的小类
#         · 统计方阵：任意两个小类共存的次数
#         · 这个方阵取反就是个不连通图，划分出每个子图即可
#     2. 如果大类足够小，发现给定用哪几个大类就能确定怎么调，就直接用大类取代小类，decoder直接一步给定所有大类的配方
#     3. 否则，decoder还要能从大类中选出小类，或者人工细分大类

pth = r"D:\Users\DELL\Desktop\datasets\cocktail\jsons\all.json"
tgt = r"D:\Users\DELL\Desktop\datasets\cocktail\jsons\all_filter.jsonl"

dict_pth = r"D:\Users\DELL\Desktop\datasets\cocktail\classes\name2class.csv"

data = pd.read_csv(dict_pth, encoding='gbk', usecols=[1, 2])
dic = {}

for i, (name, cls) in data.iterrows():
    dic[name] = cls

tgt_datas = []

lst = []


def filter(ct):
    if ct in dic.keys():
        return dic[ct]
    ct = ct.replace('House-made', '')
    ct = ct.replace("Liquor", 'Liqueur')
    ct = ct.replace("Whisky", 'Whiskey')
    ct = ct.replace('Rhum', 'Rum')
    ct = ct.strip()
    if 'tonic' in ct.lower():
        ct = 'Tonic Water'
    if 'oz' == ct[:2]:
        ct = ct[2:].strip()
    elif ct[0] in "1234567890":
        return None
    elif 'top' == ct[:3]:
        ct = ct[3:].strip()
    elif ct[-1] == '*':
        while ct[-1] == '*':
            ct = ct[:-1]
    elif ct[:5].lower() == 'fresh':
        ct = ct[5:].strip()
    if 'vodka' in ct.lower():
        if 'karlsson' in ct.lower() and 'infused' not in ct.lower():
            ct = "Karlsson's Gold Vodka"
        elif 'hophead' in ct.lower():
            ct = 'Hophead Vodka'
        elif 'absolut' in ct.lower() or 'Vodka' == ct:
            ct = 'Vodka'
        elif 'hangar' in ct.lower():
            ct = 'Hangar 1 Vodka'
        else:
            pass
    elif 'lemon' in ct.lower():
        if 'lemon juice' in ct.lower() or 'LemonJuice' == ct or 'Lemon Sour' == ct:
            ct = 'Lemon Juice'
        elif 'twist' in ct.lower() or 'zest' in ct.lower() or 'peel' in ct.lower():
            ct = 'Lemon Peel'
        elif 'bitter' in ct.lower():
            ct = 'Lemon Bitters'
        elif 'syrup' in ct.lower():
            ct = 'Lemon Syrup'
        elif 'lemonade' in ct.lower() or 'soda' in ct.lower() or 'sparkling' in ct.lower():
            ct = 'Lemon Soda'
        elif 'curd' in ct.lower() or 'sorbet' in ct.lower():
            ct = "Lemon Sorbet"
        elif 'verbana' in ct.lower() or 'verbena' in ct.lower():
            ct = 'Lemon Verbana'
        elif 'green' in ct.lower():
            ct = 'Green Lemon'
        elif 'foam' in ct.lower():
            ct = 'Lemon Foam'
        else:
            ct = 'Lemon'
    elif 'syrup' in ct.lower() or ct == 'Sugar Rich Simple' or 'Orgeat' == ct:
        ct = ct.replace('House-made', '').replace('Organic', '').strip()
        if 'honey' in ct.lower():
            ct = 'Honey'
        elif 'gum' in ct.lower() or 'gomme' in ct.lower():
            ct = 'Gum Syrup'
        elif 'cane' in ct.lower():
            ct = 'Cane Syrup'
        elif 'maple' in ct.lower():
            ct = 'Maple Syrup'
        elif 'simple' in ct.lower() or 'Sugar Syrup' == ct or 'spoon Sugar Syrup' == ct or 'jalapeño' in ct.lower()\
                or 'Doug Fir Riesling Syrup' == ct:
            ct = 'Simple Syrup'
        elif 'pineapple' in ct.lower():
            ct = 'Pineapple Syrup'
        elif 'cola' in ct.lower():
            ct = 'Cola Syrup'
        elif 'cherry' in ct.lower():
            ct = 'Cherry Syrup'
        elif 'coffee' in ct.lower():
            ct = 'Coffee Syrup'
        elif 'pear' in ct.lower():
            pass
        elif 'ginger' in ct.lower():
            if 'cinnamon' in ct.lower():
                ct = 'Ginger-Cinnamon Syrup'
            elif ct == 'Finely chopped drained stem ginger in syrup':
                ct = 'Ginger Syrup'
            else:
                pass
        elif 'cinnamon' in ct.lower():
            if 'B.G.' in ct or ct == 'Cinnamon Syrup':
                ct = 'Cinnamon Syrup'
            elif 'Chipotle' in ct or 'Cinnamon syrup' == ct:
                ct = 'Cinnamon syrup'
            else:
                pass
        elif 'berry' in ct.lower():
            pass
        elif 'flower' in ct.lower():
            pass
        elif 'demerara' in ct.lower():
            ct = 'Demerara Syrup'
        elif 'sugar' in ct.lower():
            ct = 'Brown Sugar Syrup'
        elif 'pilon' in ct.lower():
            ct = 'Pilon Syrup'
        elif 'leaf' in ct.lower():
            ct = 'Bay Leaf Syrup'
        elif 'vanilla' in ct.lower():
            pass
        elif 'fruit' in ct.lower() or 'Luxardo Orgeat Syrup' == ct or 'Rhubarb Syrup' == ct:
            if 'passion' in ct.lower():
                ct = 'Passion Fruit Syrup'
            else:
                ct = 'Rhubarb Syrup'
        elif 'tonic' in ct.lower():
            pass
        elif 'orange' in ct.lower():
            ct = 'Orange Syrup'
        elif 'lavender' in ct.lower() or 'lavander' in ct.lower():
            pass
        elif 'spice' in ct.lower():
            pass
        elif 'agave' in ct.lower():
            ct = 'Agave Syrup'
        elif 'hibiscus' in ct.lower():
            pass
        elif 'tea' in ct.lower() or 'Earl Grey Syrup' == ct:
            ct = 'Tea Syrup'
        elif 'orgeat' in ct.lower():  # 杏仁
            ct = 'Orgeat syrup'
        elif 'orchard' in ct.lower():  # 苹果
            pass
        else:
            pass
    elif 'juice' in ct.lower() or 'Lime' == ct or 'Lime Sour' == ct:
        ct = ct.replace('Squeezed', '').replace('Unfiltered', '').strip()
        if 'lime' in ct.lower():
            ct = 'Lime juice'
        elif 'orange' in ct.lower():
            ct = 'Orange Juice'
        elif 'berry' in ct.lower():
            if 'cranberry' in ct.lower():
                ct = 'Cranberry Juice'
            elif 'strawberry' in ct.lower():
                ct = 'Strawberry Juice'
        elif 'juiced' == ct:
            ct = 'Juice'
        else:
            pass
    elif 'soda' in ct.lower():
        if 'soda water' in ct.lower() or ct == 'Soda' or 'club soda' in ct.lower() or 'Fever-Tree Soda' == ct:
            ct = 'Soda'
        elif 'grapefruit' in ct.lower():
            ct = 'Grapefruit Soda'
        elif 'almond' in ct.lower():
            pass
        else:
            ct = 'Soda'
    elif 'gin' in ct.lower()[-5:]:
        if 'junipero' in ct.lower():
            pass
        elif 'london' in ct.lower() or 'broker' in ct.lower():
            ct = 'London Dry Gin'
        elif 'old tom' in ct.lower():
            pass
        elif 'plymouth' in ct.lower():
            # print(ct)
            pass
        elif 'infused' in ct.lower():
            pass
        elif 'dry' in ct.lower():
            pass
        elif 'genevieve' in ct.lower():
            pass
        elif 'Gin' == ct:
            pass
        else:
            pass
    elif 'bitter' in ct.lower() or 'Mole' == ct:
        if 'chocolate' in ct.lower():
            ct = 'Chocolate Bitters'
        elif 'banana' in ct.lower() or 'tiki' in ct.lower():
            ct = 'Roasted Banana Bitters'
        elif 'luxardo' in ct.lower():
            ct = 'Luxardo Bitter'
        elif 'orange' in ct.lower():
            ct = 'Orange Bitters'
        elif 'angostura' in ct.lower():
            ct = ct.replace('(optional)', '').strip()
        elif 'Bitter' == ct or 'Bitters' == ct:
            ct = 'Bitters'
        elif 'tempus' in ct.lower():
            ct = "Tempus Fugit Abbott's Bitters"
        elif 'cherry' in ct.lower():
            ct = 'Cherry Bitters'
        elif 'pear' in ct.lower():
            pass
        elif 'bittercube' in ct.lower():
            if 'jamaican' in ct.lower():
                ct = 'Bittercube Bitters Jamaican'
        elif 'mole' in ct.lower():
            ct = 'Mole Bitters'
        elif 'bittermens' in ct.lower():
            if 'elemakule' in ct.lower():
                ct = 'Bittermens Elemakule Tiki Bitters'
            else:
                pass
        elif 'peychaud' in ct.lower():
            ct = "Peychaud's Bitters"
        elif 'apple' in ct.lower():
            ct = 'Apple Bitters'
        elif 'celery' in ct.lower():
            pass
        elif 'fruit' in ct.lower():
            ct = 'Grapefruit Bitters'
        elif 'boker' in ct.lower():
            ct = 'Boker’s Bitters'
        elif 'berry' in ct.lower():
            pass
        elif 'aromatic' in ct.lower():
            ct = 'Aromatic Bitters'
        elif 'whiskey' in ct.lower():
            ct = 'Whiskey Barrel Bitters'
        elif 'fee' in ct.lower() or 'rhubarb' in ct.lower() or 'rubarb' in ct.lower():
            if 'rhubard' in ct.lower() or 'rhubarb' in ct.lower() or 'rubarb' in ct.lower():
                ct = 'Fee Brothers Rhubard bitters'
            else:
                pass
        else:
            pass
    elif 'luxardo' in ct.lower() or 'liqueur' in ct.lower() or "King's Ginger" in ct:
        # print(ct)
        if 'cherry' in ct.lower():
            ct = 'Cherry Liqueur'
        elif 'apple' in ct.lower():
            ct = 'Apple Liqueur'
        elif 'chocolate' in ct.lower():
            ct = 'Chocolate Liqueur'
        elif 'orange' in ct.lower():
            ct = 'Orange Liqueur'
        elif 'honey' in ct.lower():
            ct = 'Honey Liqueur'
        elif 'ginger' in ct.lower():
            ct = 'Ginger Liqueur'
        elif 'pear' in ct.lower():
            ct = 'Pear Liqueur'
        elif 'watermelon' in ct.lower():
            ct = 'Watermelon Liqueur'
        elif 'hazelnut' in ct.lower():
            ct = 'Hazelnut Liqueur'
        elif 'Liqueur de Violettes' in ct:
            ct = 'Liqueur de Violettes'
        elif 'Luxardo Anisette' in ct:
            ct = 'Luxardo Anisette'
        elif 'Luxardo Amaretto' in ct:
            ct = 'Luxardo Amaretto'
        elif 'Luxardo Triplum' in ct:
            ct = 'Luxardo Triplum'
        elif 'Luxardo Maraschino' in ct:
            ct = 'Luxardo Maraschino'
        elif 'Luxardo Slivovitz' in ct:
            ct = 'Luxardo Slivovitz'
        elif 'Luxardo Fernet' in ct:
            ct = 'Luxardo Fernet'
        elif 'Luxardo Limoncello' in ct:
            ct = 'Luxardo Limoncello'
        else:
            pass
    elif 'whiskey' in ct.lower() or 'bourbon' in ct.lower() or ct == 'Glenrothes Vintage Reserve':
        if 'rye' in ct.lower() or 'bourbon' in ct.lower():
            ct = 'Bourbon Whiskey'
        else:
            pass
    elif 'rum' in ct.lower():
        if 'pink' in ct.lower() or 'pigeon' in ct.lower():
            ct = 'Pink Pigeon Rum'
        elif 'white' in ct.lower():
            ct = 'White Rum'
        elif 'black' in ct.lower():
            ct = 'Black Rum'
        elif 'aged' in ct.lower() or 'old' in ct.lower():
            ct = 'Aged Rum'
        elif 'light' in ct.lower():
            ct = 'Light Rum'
        elif 'english' in ct.lower():
            ct = 'English Harbour Rum'
        elif 'bank' in ct.lower():
            ct = 'Banks 5 Rum'
        elif 'bacardi' in ct.lower():
            ct = 'Bacardi Superior Rum'
        elif 'pusser' in ct.lower():
            ct = "Pusser's Rum"
        elif 'gossling' in ct.lower():
            ct = "Gossling's Rum"
        else:
            pass
    elif ('wine' in ct.lower() and 'brandy' not in ct.lower()) or ct == 'White dry white(preferably Sauvignon)':
        if 'sparkling' in ct.lower():
            ct = 'Sparkling Wine'
        elif 'dry' in ct.lower():
            if 'white' in ct.lower():
                ct = 'Dry White Wine'
        elif 'red' in ct.lower():
            pass
        elif 'aperitif' in ct.lower():
            ct = 'Aperitif Wine'
        else:
            pass
    elif 'vermouth' in ct.lower():
        if 'dry' in ct.lower():
            ct = 'Dry vermouth'
        elif 'tempus fugit' in ct.lower():
            ct = 'Tempus Fugit Alessio Vermouth'
        elif 'sweet' in ct.lower():
            ct = 'Sweet Vermouth'
        elif 'blanc' in ct.lower():
            ct = 'Blanco Vermouth'
        elif 'red' in ct.lower() or 'antica' in ct.lower():
            ct = 'Red Vermouth'
        elif 'dolin' in ct.lower():
            ct = 'Dolin Vermouth'
        else:
            pass
    elif 'mezcal' in ct.lower():
        pass
    elif 'tonic' in ct.lower():
        pass
    elif 'pisco' in ct.lower():
        if 'barsol' in ct.lower():
            ct = 'BarSol Pisco'
        else:
            pass
    elif 'brandy' in ct.lower():
        pass
    elif 'white' in ct.lower():
        if ct.lower() == 'white':
            return None
        elif 'egg' in ct.lower():
            ct = 'Egg white'
        elif 'peach' in ct.lower():
            ct = 'White peach puree'
        elif 'Peel (white pith removed and julienned)' == ct:
            return None
        else:
            pass
    elif 'water' in ct.lower():
        if 'sparkling' in ct.lower() or 'carbonated' in ct.lower():
            ct = 'Soda'
        elif 'hot' in ct.lower() or 'cold' in ct.lower() or 'spring' in ct.lower() or 'mineral' in ct.lower() \
                or 'boiling' in ct.lower():
            ct = 'Water'
        elif 'watermelon' in ct.lower():
            pass
        elif 'coconut' in ct.lower():
            pass
        elif 'orange' in ct.lower():
            ct = 'Orange Flower Water'
        else:
            pass
    elif 'ginger' in ct.lower():
        if 'beer' in ct.lower() or 'ale' in ct.lower() or 'brew' in ct.lower():
            ct = 'Ginger beer'
        elif 'pieces' in ct.lower() or 'snaps' in ct.lower() or ct == 'ginger' or 'hunk' in ct.lower():
            ct = 'Ginger'
        elif 'root' in ct.lower() or 'brown' in ct.lower():
            ct = 'Ginger Root'
        else:
            pass
    elif 'beer' in ct.lower():
        pass
    elif 'scotch' in ct.lower():
        pass
    elif 'H by HINE' == ct or 'cognac' in ct.lower():
        ct = 'Cognac'
    elif 'sugar' in ct.lower():
        if 'demerara' in ct.lower() or 'cane' in ct.lower():
            ct = 'Cane Sugar'
        elif 'caster' in ct.lower() or ct == 'Sugar':
            ct = 'Sugar'
        else:
            pass
    elif 'leaves' in ct.lower() or 'leaf' in ct.lower():
        if ct == 'leaves' or ct == 'Leaves' or ct == 'Leaf' or ct == 'leaf':
            ct = 'Leaves'
        else:
            pass
    elif 'salt' in ct.lower():
        pass
    elif 'milk' in ct.lower() and 'chocolate' not in ct.lower() and 'coffee' not in ct.lower():
        if 'coconut' in ct.lower():
            ct = 'Coconut Milk'
        elif 'condense' in ct.lower():
            ct = 'Condensed Milk'
        else:
            ct = 'Milk'
    elif 'chocolate' in ct.lower():
        if 'milk' in ct.lower():
            ct = 'Milk Chocolate'
        elif 'dark' in ct.lower():
            ct = 'Dark Chocolate'
        else:
            pass
    elif 'coffee' in ct.lower() and 'infused' not in ct.lower():
        if 'cold' in ct.lower():
            ct = 'Cold Coffee'
        elif 'hot' in ct.lower():
            ct = 'Hot Coffee'
        elif 'espresso' in ct.lower():
            ct = 'Espresso Coffee'
        else:
            pass
    elif 'gum' in ct.lower():
        pass
    elif 'tequila' in ct.lower():
        if 'reposado' in ct.lower():   # 微陈龙舌兰
            ct = 'Reposado Tequila'
        elif 'blanco' in ct.lower():   # 白龙舌兰
            ct = 'Blanco Tequila'
        elif 'anejo' in ct.lower():    # 百年陈龙舌兰
            ct = 'Anejo Tequila'
        else:
            pass
    elif 'branca' in ct.lower():   # 两种利口酒
        pass
    elif 'agave' in ct.lower():
        if 'nectar' in ct.lower():
            pass
        else:
            pass
    elif 'blanc' in ct.lower() and 'blanche' not in ct.lower():   # 干白
        ct = 'Blanc'
    elif 'laphroaig' in ct.lower():
        pass
    elif 'chartreuse' in ct.lower():
        pass
    elif 'classico' in ct.lower():
        pass
    elif 'old vine zin' in ct.lower():
        pass
    elif 'apple' in ct.lower():
        if 'cider' in ct.lower():
            ct = 'Apple Cider'
        elif 'pineapple' in ct.lower():
            if 'sorbet' in ct.lower():
                ct = 'Pineapple Sorbet'
            else:
                 ct = 'Pineapple'
        elif 'pur' in ct.lower():
            ct = 'Apple Puree'
        else:
            pass
    elif 'fruit' in ct.lower():
        if 'passion' in ct.lower():
            ct = 'Passion Fruit'
        elif 'grapefruit' in ct.lower():
            if 'sorbet' in ct.lower():
                ct = 'Grapefruit Sorbet'
            else:
                ct = 'Grapefruit'
        else:
            pass
    elif 'vinegar' in ct.lower():
        pass
    elif 'pear' in ct.lower():
        pass
    elif 'egg' in ct.lower():
        pass
    elif 'germain' in ct.lower():
        pass
    elif 'cream' in ct.lower():
        if 'coconut' in ct.lower():
            ct = 'Coconut Cream'
        elif 'vanilla' in ct.lower():
            ct = 'Vanilla Ice Cream'
        elif 'marshmallow' in ct.lower():
            ct = 'Marshmallow Whip Cream'
        elif 'orange' in ct.lower():
            ct = 'Orange Cream Citrate'
        else:
            ct = 'Cream'
    elif 'sherry' in ct.lower():
        ct = 'Sherry'
    elif 'spark' in ct.lower():
        if 'rose' in ct.lower():
            ct = 'Sparkling Rose'
    elif 'cherry' in ct.lower():
        if 'muddled' in ct.lower() or ct == 'Cherry':
            ct = 'Cherry'
        else:
            pass
    elif 'lime' in ct.lower():
        if 'sweet' in ct.lower():
            ct = "Lime Sweet n' Sour mix"
        else:
            ct = 'Lime'
    elif 'champagne' in ct.lower():
        ct = 'Champagne'
    elif 'berry' in ct.lower():
        if 'strawberry' in ct.lower():
            if 'shrub' in ct.lower():
                pass
            else:
                ct = 'Strawberry'
        else:
            pass
    elif 'superieure' in ct.lower():   # 苦艾
        ct = 'superieure'
    elif 'solera' in ct.lower():
        pass
    elif 'reserve' in ct.lower():
        pass
    else:
        pass

    if ct in dic.keys():
        ct = dic[ct]
    else:
        return None
    lst.append(ct)
    return ct

if __name__ == '__main__':

    with open(pth, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    for line in datas:

        rec = {
            'name': line['name'],
            'contents': {}
        }

        for ctt in line['contents'].keys():
            if ctt == 'others':
                for ct in line['contents']['others']:
                    ct = filter(ct)
                    if ct is not None:
                        rec['contents'][ct] = None
            else:
                ctt_ = ctt
                ctt = filter(ctt)
                if ctt is not None:
                    rec['contents'][ctt] = line['contents'][ctt_]

        tgt_datas.append(rec)

    with open(tgt, 'w', encoding='utf-8') as f:
        for data in tgt_datas:
            f.write(json.dumps(data))
            f.write('\n')

