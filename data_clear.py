from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


def get_name_titles(df):
    # 根据Name中的称谓提取身份
    titles = set()
    for name in df['Name']:
        titles.add(name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }
    df['Title'] = df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    df['Title'] = df.Title.map(Title_Dictionary)
    return df


def fill_missing_fare(df):
    # 测试集中存在缺失的fare
    fare = df['Fare']
    avg_fare = np.mean(fare)
    df.loc[df.Fare.isnull(), 'Fare'] = avg_fare
    return df


def fill_missing_ages(df):
    # Age 运用随机森林拟合，根据已知年龄组的信息训练模型，回归拟合出未知年龄
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=1)
    rfr.fit(x, y)

    predicted_age = rfr.predict(unknown_age[:, 1:])

    df.loc[df.Age.isnull(), 'Age'] = predicted_age

    return df


def fill_missing_embarked(df):
    # Embarked 用众数填充(因为只有两个缺失值，且特征为类别特征)
    embarked_df = df['Embarked']

    known_embarked = embarked_df[embarked_df.notnull()].as_matrix()

    count = {}
    for item in known_embarked:
        if item not in count:
            count[item] = 1
        else:
            count[item] += 1

    max_value = max(count.values())
    for k, v in count.items():
        if v == max_value:
            most_embarked = k
            break

    df.loc[df.Embarked.isnull(), 'Embarked'] = most_embarked
    return df


def fill_missing_data(df):
    df = fill_missing_fare(df)
    df = fill_missing_embarked(df)
    df = fill_missing_ages(df)
    return df


def one_hot_encode(df):
    # cabin、embarked、sex、pclass、Title标签型特征做one_hot编码
    # cabin先按有无分成两类，再one_hot编码
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    dummies_cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_sex = pd.get_dummies(df['Sex'], prefix='Sex')
    dummies_pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    dummies_title = pd.get_dummies(df['Title'], prefix='Title')

    df = pd.concat([df, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass, dummies_title], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title'], axis=1, inplace=True)
    return df


def make_normalize(df):
    # age、fare、SibSp、Parch、FamilySize等数值型特征做归一化
    std = StandardScaler()

    df['Age'] = std.fit_transform(df['Age'])
    df['Fare'] = std.fit_transform(df['Fare'])
    df['SibSp'] = std.fit_transform(df['SibSp'])
    df['Parch'] = std.fit_transform(df['Parch'])
    df['FamilySize'] = std.fit_transform(df['FamilySize'])
    return df


def get_family(df):
    # 根据家庭人数(算上自己)，手工划分Size
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
    return df


def feature_extract(df):
    # 进行特征提取
    df = get_family(df)
    df = one_hot_encode(df)
    df = make_normalize(df)
    return df


def data_clear(df):
    df = get_name_titles(df)
    df = fill_missing_data(df)
    df = feature_extract(df)
    return df


if __name__ == '__main__':
    # 读取训练集
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    train.drop(['PassengerId'], axis=1, inplace=True)
    # 数据处理
    train = data_clear(train)
    test = data_clear(test)
    test['Title_Royalty'] = pd.DataFrame(np.zeros((418, 1)))
    # 写入数据集
    train.to_csv('input/train_input.csv', sep=',', header=True, index=False)
    test.to_csv('input/test_input.csv', sep=',', header=True, index=False)


