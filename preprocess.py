import csv
import os
import pickle

import pandas as pd
import json
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def numerize(data, user2id, item2id):
    uid = list(map(lambda x: user2id[x], data['user_id']))
    iid = list(map(lambda x: item2id[x], data['item_id']))
    data['user_id'] = uid
    data['item_id'] = iid
    return data


def get_count(data, id):
    ids = set(data[id].tolist())
    return ids


def extract(data_dict):
    x = []
    y = []
    for i in data_dict.values:
        uid = i[0]
        iid = i[1]
        x.append([uid, iid])
        y.append(float(i[2]))
    return x, y


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"what's", "what is ", string)
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    reviews_path = r"./yelp_dataset/yelp_academic_dataset_review.json"
    # read the data
    print(f"{now()}: Step1: loading raw review datasets...")
    file = open(reviews_path, 'r', encoding="utf-8")

    users_id = []
    items_id = []
    ratings = []
    reviews = []
    dates = []

    length = 10000
    i = 0

    for line in file:
        js = json.loads(line)
        # print(js)
        if str(js['user_id']) == 'unknown':
            print("unknown user id")
            continue
        if str(js['business_id']) == 'unknown':
            print("unkown item id")
            continue
        date = str(js["date"])
        _date = date.split('-')[0]

        if _date == '2018':
            reviews.append(js['text'])
            users_id.append(str(js['user_id']))
            items_id.append(str(js['business_id']))
            ratings.append(str(js['stars']))
            dates.append(date)
            i += 1
            if i == length:
                break

    data_frame = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id),
                  'ratings': pd.Series(ratings), 'reviews': pd.Series(reviews), 'date': pd.Series(dates)}
    data = pd.DataFrame(data_frame)
    # print(data)

    # # 将u_id和i_id进行唯一编码
    # userID = list(set(users_id))
    # itemID = list(set(items_id))
    #
    # # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    # item2id = {key: index for index, key in enumerate(itemID)}
    # user2id = {key: index for index, key in enumerate(userID)}
    # data = numerize(data, user2id, item2id)

    # 将data[ratings]转换为int类型
    data['ratings'] = data['ratings'].astype(float).astype(int)

    # clean the reviews
    for i in data.values:
        str_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))

        if len(str_review.strip()) == 0:
            str_review = "<unk>"

        # 在data中把i[3]替换为str_review
        data.replace(i[3], str_review, inplace=True)

    uidList, iidList = get_count(data, 'user_id'), get_count(data, 'item_id')
    userNum_all = len(uidList)
    itemNum_all = len(iidList)
    print("===============Start:all  rawData size======================")
    print(f"dataNum: {data.shape[0]}")
    print(f"userNum: {userNum_all}")
    print(f"itemNum: {itemNum_all}")
    print(f"data densiy: {data.shape[0] / float(userNum_all * itemNum_all):.4f}")

    # 将data储存为csv文件
    data.to_csv('./dataset/yelp2018.csv', index=False)

    print("===============End: rawData size========================")
    #
    print(f"-" * 60)
    print(f"{now()} Step2: split datsets into train/test, save into data")
    data = data[data['ratings'] >= 3]
    print(len(data))

    data.groupby(['ratings'])['ratings'].count()

    train, test = train_test_split(data.values, test_size=0.2, random_state=42)
    train = pd.DataFrame(train, columns=data.columns)
    test = pd.DataFrame(test, columns=data.columns)
    print("Train Size  : ", len(train))
    print("Test Size : ", len(test))

    # Label Encoding the User and Item IDs
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()
    train['user_id_idx'] = le_user.fit_transform(train['user_id'].values)
    train['item_id_idx'] = le_item.fit_transform(train['item_id'].values)

    train_user_ids = train['user_id'].unique()
    train_item_ids = train['item_id'].unique()

    print(len(train_user_ids), len(train_item_ids))

    test = test[(test['user_id'].isin(train_user_ids)) & (test['item_id'].isin(train_item_ids))]
    # 删去test中reviews列，防止信息泄露
    test = test.drop(['reviews'], axis=1)
    print("Test Size After processing: ", len(test))

    test['user_id_idx'] = le_user.transform(test['user_id'].values)
    test['item_id_idx'] = le_item.transform(test['item_id'].values)

    n_users = train['user_id_idx'].nunique()
    n_items = train['item_id_idx'].nunique()
    print("Number of Unique Users : ", n_users)
    print("Number of unique Items : ", n_items)

    print(test.head(10))

    # get the reviews for each user_id_idx in train data
    user_reviews = []
    for i in range(train['user_id_idx'].nunique()):
        user_reviews = train[train['user_id_idx'] == i]['reviews'].tolist()
        user_reviews.append([i] + user_reviews)
    # save as csv file
    with open('./dataset/user_reviews.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(user_reviews)

    # get the reviews for each item_id_idx in train data
    user_reviews = []
    for i in range(train['item_id_idx'].nunique()):
        user_reviews = train[train['item_id_idx'] == i]['reviews'].tolist()
        user_reviews.append([i] + user_reviews)
    # save as csv file
    with open('./dataset/item_reviews.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(user_reviews)

    map_list = []
    for i in range(train['user_id_idx'].nunique()):
        to_item = train[train['user_id_idx'] == i]['item_id_idx'].tolist()
        map_list.append([i] + to_item)
        # map_list.extend(train[train['user_id_idx'] == i]['item_id_idx'].tolist())

    # 储存为txt文件
    with open('dataset/train.txt', 'w') as f:
        for i in map_list:
            # 逐个写入i中的元素
            a = 0
            for j in i:
                i_len = len(i)
                f.write(str(j))
                f.write(' ') if a != i_len - 1 else f.write('\n')
                a += 1

    # 提取test后两列
    test_list = test[['user_id_idx', 'item_id_idx']]
    test_list.head()
    # 储存为txt文件
    test_list.to_csv('dataset/test.txt', index=False, header=False, sep=' ')
    print("Step2 Done!")

    print(f"-" * 60)
    print(f"{now()} Step3: get the review for each user/item")

    # get the reviews for each user_id_idx in train data
    users_reviews = []
    for i in range(train['user_id_idx'].nunique()):
        # print(i)
        user = train[train['user_id_idx'] == i]['reviews'].tolist()
        # print(user)
        users_reviews.append([i] + user)
        # print(users_reviews)

    # save as csv file
    with open('./dataset/user_reviews.csv', 'w', newline='', errors='ignore') as f:
        writer = csv.writer(f)
        writer.writerows(users_reviews)
    print(f"save user_reviews.csv done!")

    # get the reviews for each item_id_idx in train data
    item_reviews = []
    # print(train['item_id_idx'].nunique())
    for i in range(train['item_id_idx'].nunique()):
        # print(i)
        item = train[train['item_id_idx'] == i]['reviews'].tolist()
        item_reviews.append([i] + item)

    # save as csv file
    with open('./dataset/item_reviews.csv', 'w', newline='', errors='ignore') as f:
        writer = csv.writer(f)
        writer.writerows(item_reviews)
    print(f"save item_reviews.csv done!")

    print(f"Step3 Done!")
