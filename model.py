import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_data = pd.read_csv('input/train_input.csv')
    y = train_data['Survived']
    X = train_data.values[:, 1:]

    # # 本地交叉验证，多种分类器
    # models_name = ['DecisionTree', 'GaussianNB', 'SVM', 'RF']
    # models = [DecisionTreeClassifier(), GaussianNB(), SVC(),
    #           RandomForestClassifier(n_estimators=500, max_features='sqrt')]
    # RF = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    #
    # # num作为标号，方便文件写入记录
    # record = []
    # for name, model in zip(models_name, models):
    #     evaluate_list = ['accuracy', 'f1', 'roc_auc']
    #     for item in evaluate_list:
    #         score = cross_val_score(model, x, y, scoring=item, cv=5).mean()
    #         record.append(score)
    #         # print(name + ' ' + item + ' is :' + str(score))
    #         print('%s %s score is: %f' % (name, item, score))

    # RF调参
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV
    # rf1 = RandomForestClassifier(random_state=10)
    # rf1.fit(X, y)
    # # print(RF0.oob_score_)
    # list = ['accuracy', 'f1', 'roc_auc']
    # for item in list:
    #     score = cross_val_score(rf1, X, y, scoring=item, cv=10).mean()
    #     print(score)

    # # ----------------------------网格调参1----------------------------
    # # best_params：{'n_estimators': 50}
    # # best_score: 0.8570522593714579
    # # 弱学习器的最大迭代次数，或者说最大的弱学习器的个数。一般来说n_estimators太小，容易欠拟合，n_estimators太大，计算量会太大，
    # # 并且n_estimators到一定的数量后，再增大n_estimators获得的模型提升会很小，所以一般选择一个适中的数值。默认是100
    # param_test1 = {'n_estimators': range(45, 60, 1)}
    # gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
    #                                                          min_samples_leaf=20, max_depth=8, max_features='sqrt',
    #                                                          random_state=10),
    #                         param_grid=param_test1, scoring='roc_auc', cv=5)
    # gsearch1.fit(X, y)
    # # for item in gsearch1.cv_results_:
    # #     print(item, ':', gsearch1.cv_results_[item])
    # print(gsearch1.best_params_)
    # print(gsearch1.best_score_)

    # # ----------------------------网格调参2----------------------------
    # # best_params: {'max_depth': 10, 'min_samples_split': 42}
    # # best_score: 0.8607364655257614
    # # 最大深度max_depth
    # # 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split，
    # # 则不会继续再尝试选择最优特征来进行划分。 默认是2.如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。
    # param_test2 = {'max_depth': range(7, 13, 1), 'min_samples_split': range(40, 55, 1)}
    # gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=50,
    #                                                          min_samples_leaf=20, max_features='sqrt', oob_score=True,
    #                                                          random_state=10),
    #                         param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
    # gsearch2.fit(X, y)
    # # for item in gsearch2.cv_results_:
    # #     print(item, ':', gsearch2.cv_results_[item])
    # print(gsearch2.best_params_)
    # print(gsearch2.best_score_)

    # # ----------------------------网格调参3 -----------------------------
    # # best_params：{'min_samples_leaf': 1, 'min_samples_split': 17}
    # # best_score: 0.8778384018509314
    # # 内部节点再划分所需最小样本数min_samples_split: 这个值限制了子树继续划分的条件，
    # # 如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分。
    # # 叶子节点最少样本数min_samples_leaf: 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。
    # param_test3 = {'min_samples_split': range(5, 30, 1), 'min_samples_leaf': range(1, 3, 1)}
    # gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=50, max_depth=13,
    #                                                          max_features='sqrt', oob_score=True, random_state=10),
    #                         param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
    # gsearch3.fit(X, y)
    # # for item in gsearch3.cv_results_:
    # #     print(item, ':', gsearch3.cv_results_[item])
    # print(gsearch3.best_params_)
    # print(gsearch3.best_score_)

    # # ----------------------------网格调参4 -----------------------------
    # # best_params：{'max_features': 2}
    # # best_score: 0.857750656450708
    # param_test4 = {'max_features': range(1, 10, 1)}
    # gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=50, max_depth=13, min_samples_split=120,
    #                                                          min_samples_leaf=20, oob_score=True, random_state=10),
    #                         param_grid=param_test4, scoring='roc_auc', iid=False, cv=5)
    # gsearch4.fit(X, y)
    # # for item in gsearch4.cv_results_:
    # #     print(item, ':', gsearch4.cv_results_[item])
    # print(gsearch4.best_params_)
    # print(gsearch4.best_score_)

    # 参数整合
    # ,oob_score=True
    rf2 = RandomForestClassifier(n_estimators=50, max_features=2, min_samples_leaf=1, oob_score=True,
                                min_samples_split=42, max_depth=10, random_state=10)
    rf2.fit(X, y)
    # print(rf2.oob_score_)
    # list = ['accuracy', 'f1', 'roc_auc']
    # for item in list:
    #     score = cross_val_score(rf2, X, y, scoring=item, cv=10).mean()
    #     print(score)

    # 输出
    test_data = pd.read_csv('input/test_input.csv')
    pre = rf2.predict(test_data.values[:, 1:])
    pre = pd.DataFrame(pre, index=None, columns=['Survived'])
    output = test_data['PassengerId']
    output = pd.concat([output, pre], axis=1)
    output.to_csv('./output/result3.csv', index=None)
    print('test')


    # # 绘制特征重要性图
    # clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    # clf = clf.fit(train_data, y)
    # features = pd.DataFrame()
    # features['feature'] = train_data.columns
    # features['importance'] = clf.feature_importances_
    # features.sort_values(by=['importance'], ascending=True, inplace=True)
    # features.set_index('feature', inplace=True)
    # ax = features.plot(kind='barh', figsize=(25, 25))
    # fig = ax.get_figure()
    # fig.show()
    # fig.savefig('fig.png')
