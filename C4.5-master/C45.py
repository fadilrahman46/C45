import math
from xml.dom import minidom
from xml.etree import ElementTree as ET


# beautify output ，recursive call，make the output xml looks like stages
def beautify(elem, level=0):    # elem 是一个xml对象，参考文档了解python xml对象属性操作
    i = "\n" + level * " "      # new line and indentation
    if len(elem):
        if not elem.text or not elem.text.strip(): # if the lable has no text,then new line and indentation 若标签内无text，则换行缩进
            elem.text = i + " "
        for e in elem:                             # for every sub-lable recursive call the beautify function对于每一个子标签，递归调用
            beautify(e, level+1)
        if not e.tail or not e.tail.strip():       
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem


# if attr is_num
def is_num(attr):
    for x in set(attr):
        if not x == "?":
            try:
                x = float(x)
                return isinstance(x, float)
            except ValueError:
                return False
    return True


# calculate entropy Entropy(x) = -Σp(x=i)log(p(x=i),2) i∈x i≠"?"
def entropy(x):
    ent = 0
    for k in set(x):
        p_i = float(x.count(k)) / len(x)
        ent -= p_i * math.log(p_i, 2)
    return ent


# !!! For distributed set !!!
# ###########################
# calculate information gain ratio  ratio = gain(attr) / entropy(attr)
# filter  attr and cate => att and cat 
# entropy(att)
# gain(att) = entropy(cat) - Σ p(attr = i)entropy(cat_i)  
def gain_ratio(category, attr):
    s = 0
    cat = []
    att = []
    for i in range(len(attr)):                  # filter  attr and cate => att and cat 
        if not attr[i] == "?":
            cat.append(category[i])
            att.append(attr[i])
    for i in set(att):                          # s = Σ p(attr = i)entropy(cat_i)  
        p_i = float(att.count(i)) / len(att)   
        cat_i = []
        for j in range(len(cat)):              
            if att[j] == i:
                cat_i.append(cat[j])
        s += p_i * entropy(cat_i)               
    gain = entropy(cat) - s                     # gain(att) = entropy(cat) - Σ p(attr = i)entropy(cat_i)  
    ent_att = entropy(att)                      # entropy(attr)
    if ent_att == 0:
        return 0
    else:
        return gain / ent_att
    
    
# !!! For continuous set !!!
# ###########################
# calculate information gain ratio  ratio = gain(attr) / entropy(attr)
# filter  attr and cate => att and cat 
# sort 
# gain(att) = entropy(cat) - min(E | E = -p(att<i)entropy(cat<i) -p(att>=i)entropy(cat>=i)   
# p(att>=i) = 1 - p(att<i)
# entropy(att)
def gain(category, attr):
    cats = []
    for i in range(len(attr)):                              # filter  attr and cate => att and cat 
        if not attr[i] == "?":                              # sorted
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])

    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]

    if len(set(att)) == 1:
        return 0
    else:
        gains = []
        div_point = []
        for i in range(1, len(cat)):
            if not att[i] == att[i-1]:
                # min(E | E = -p(att<i)entropy(cat<i) -p(att>=i)entropy(cat>=i) 
                gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
                div_point.append(i)
        # gain(att) = entropy(cat) - min(E | E = -p(att<i)entropy(cat<i) -p(att>=i)entropy(cat>=i)
        gain = entropy(cat) - min(gains)

        # entropy(att)
        p_1 = float(div_point[gains.index(min(gains))]) / len(cat)
        ent_attr = -p_1 * math.log(p_1, 2) - (1 - p_1) * math.log((1 - p_1), 2)
        return gain / ent_attr


# for continuous set find the division point which is the minimum in gains
# so that gain = entropy(cat) - min(gains) can be the maximum
def division_point(category, attr):
    cats = []
    for i in range(len(attr)):                              # filter  attr and cate => att and cat 
        if not attr[i] == "?":                              # sorted
            cats.append([float(attr[i]), category[i]])
    cats = sorted(cats, key=lambda x: x[0])
    # print(cats)
    cat = [cats[i][1] for i in range(len(cats))]
    att = [cats[i][0] for i in range(len(cats))]

    gains = []
    div_point = []
    for i in range(1, len(cat)):
        if not att[i] == att[i-1]:
            gains.append(entropy(cat[:i]) * float(i) / len(cat) + entropy(cat[i:]) * (1 - float(i) / len(cat)))
            div_point.append(i)
    return att[div_point[gains.index(min(gains))]]


# grow decision tree 
# recursively call this function
def grow_tree(data, category, parent, attrs_names):
    if len(set(category)) > 1:
        division = []
        for i in range(len(data)):
            if set(data[i]) == set("?"):
                division.append(0)
            else:
                if is_num(data[i]):
                    division.append(gain(category, data[i]))
                else:
                    division.append(gain_ratio(category, data[i]))
        if max(division) == 0:
            num_max = 0
            for cat in set(category):
                num_cat = category.count(cat)
                if num_cat > num_max:
                    num_max = num_cat
                    most_cat = cat
                    parent.text = most_cat
        else:
            index_selected = division.index(max(division))
            name_selected = attrs_names[index_selected]
            if is_num(data[index_selected]):
                div_point = division_point(category, data[index_selected])
                r_son_data = [[] for i in range(len(data))]
                r_son_category = []
                l_son_data = [[] for i in range(len(data))]
                l_son_category = []
                for i in range(len(category)):
                    if not data[index_selected][i] == "?":
                        if float(data[index_selected][i]) < float(div_point):
                            l_son_category.append(category[i])
                            for j in range(len(data)):
                                l_son_data[j].append(data[j][i])
                        else:
                            r_son_category.append(category[i])
                            for j in range(len(data)):
                                r_son_data[j].append(data[j][i])
                if len(l_son_category) > 0 and len(r_son_category) > 0:
                    p_l = float(len(l_son_category)) / (len(data[index_selected]) - data[index_selected].count("?"))
                    son = ET.SubElement(parent, name_selected, {'value':str(div_point), "flag":"l","p":str(round(p_l, 3))})
                    grow_tree(l_son_data, l_son_category, son, attrs_names)
                    son = ET.SubElement(parent, name_selected, {'value':str(div_point), "flag":"r","p":str(round(1 - p_l, 3))})
                    grow_tree(r_son_data, r_son_category, son, attrs_names)
                else:
                    num_max = 0
                    for cat in set(category):
                        num_cat = category.count(cat)
                        if num_cat > num_max:
                            num_max = num_cat
                            most_cat = cat
                            parent.text = most_cat
            else:
                for k in set(data[index_selected]):
                    if not k == "?":
                        son_data = [[] for i in range(len(data))]
                        son_category = []
                        for i in range(len(category)):
                            if data[index_selected][i] == k:
                                son_category.append(category[i])
                                for j in range(len(data)):
                                    son_data[j].append(data[j][i])
                        son = ET.SubElement(parent, name_selected, {'value':k, "flag":"m", 'p':str(round(float(len(son_category))/(len(data[index_selected])-data[index_selected].count("?")),3))})
                        grow_tree(son_data, son_category, son, attrs_names)
    else:
        parent.text = category[0]


# train data and grow the tree
def train(training_obs, training_cat, xmldir):
    if not len(training_obs) == len(training_cat):
        return False
    attrs_names = training_obs[0]
    data = [[] for i in range(len(attrs_names))]
    categories = []
    for i in range(1, len(training_obs)):
        categories.append(training_cat[i])
        for j in range(len(attrs_names)):
            data[j].append(training_obs[i][j])
    root = ET.Element('DecisionTree')
    tree = ET.ElementTree(root)
    grow_tree(data, categories, root, attrs_names)
    tree.write(xmldir)
    ET.dump(beautify(root))
    return True


# 
def add(d1, d2):
    d = d1
    for i in d2:
        if i in d:
            d[i] += d2[i]
        else:
            d[i] = d2[i]
    return d


# make dicision 
def decision(root, obs, attrs_names, p):
    if root.hasChildNodes():
        att_name = root.firstChild.nodeName
        if att_name == "#text":
            return decision(root.firstChild, obs, attrs_names, p)
        else:
            att = obs[attrs_names.index(att_name)]
            if att == "?":
                d = {}
                for child in root.childNodes:
                    d = add(d, decision(child, obs, attrs_names, p * float(child.getAttribute("p"))))
                return d
            else:
                for child in root.childNodes:
                    if child.getAttribute("flag") == "m" and child.getAttribute("value") == att or \
                       child.getAttribute("flag") == "l" and float(att) < float(child.getAttribute("value")) or \
                       child.getAttribute("flag") == "r" and float(att) >= float(child.getAttribute("value")):
                        return decision(child, obs, attrs_names, p)
    else:
        return {root.nodeValue:p}


# predict the answer according to the decision tree
def predict(xmldir, testing_obs):
    doc = minidom.parse(xmldir)
    root = doc.childNodes[0]
    prediction = []
    attrs_names = testing_obs[0]
    for i in range(1, len(testing_obs)):
        answerlist = decision(root, testing_obs[i], attrs_names, 1)
        answerlist = sorted(answerlist.items(), key=lambda x: x[1], reverse=True)
        answer = answerlist[0][0]
        prediction.append(answer)
    return prediction