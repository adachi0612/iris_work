# 可視化ライブラリを読み込む
import graphviz
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns

# 教師あり学習の各手法を読み込む
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

'''
# TODO: 
# 全体的に、コメントはgoogle Docs形式にして、type hintを追加してください。
# 課題に入っているREADMEに詳細は書かれています。
# また、保守や拡張性を考えて実装することを心がけてください。kfold = KFold(n_splits=5)のところなど、数字をベタ書きするのはお勧めしません。

'''

class AnalyzeIris:
    def __init__(self):
        # load_iris()でIrisデータセットを読み込む
        self.iris = load_iris()
        # 正解ラベルのない各特徴量のデータフレームを作成
        # FIXME: dataframe はdfと略すことが多いのそちらにしてださい。
        # TODO: irisという名前も抽象度が低いので、他の名前の方がいいですが..., お任せします

        self.iris_dataframe = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )

    def get(self):
        self.iris_dataframe_label = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        # ラベルの列を追加
        self.iris_dataframe_label["Label"] = self.iris.target
        return self.iris_dataframe_label

    def get_correlation(self):
        # 各特徴量間の相関係数を求める
        return self.iris_dataframe.corr()

    def pair_plot(self, diag_kind="hist"):
        # 正解ラベルのない各特徴量のデータフレームを作成
        # FIXME: iris_dataframe_speciesは他の関数で使う予定がないなら、selfは使わなくてもいいです。
        self.iris_dataframe_species = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        # 種の正解ラベル列を追加
        self.iris_dataframe_species["Species"] = np.array(
            self.iris.target_names[i] for i in self.iris.target
        )
        # seabornでペアプロットを行う
        return sns.pairplot(
            self.iris_dataframe_species,
            hue="Species",
            diag_kind=diag_kind,
            markers="o",
            palette="tab10",
        )

    def all_supervised(self, n_neighbors=4):
        #FIXME: 二つのリストにせずに、dictにした方がいいと思います。
        model = [
            LogisticRegression(),
            LinearSVC(),
            SVC(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(n_neighbors=n_neighbors),
            LinearRegression(),
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            MLPClassifier(),
        ]
        model_name = [
            "LogisticRegression",
            "LinearSVC",
            "SVC",
            "DecisionTreeClassifier",
            "KNeighborClassifier",
            "LinearRegression",
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "MLPClassifier",
        ]
        # 5分割交差検証
        # FIXME: 5という数字を変数にして、変更しやすくするといいです。下のプリント文もこの数字に対応させてください。
        kfold = KFold(n_splits=5)
        # iris.get_supervised()での出力結果のために空のデータフレームを作る
        self.iris_supervised = pd.DataFrame()
        self.supervised_model = []

        for model, model_name in zip(model, model_name):
            print("=== {} ===".format(model_name))
            score = cross_validate(
                model,
                self.iris.data,
                self.iris.target,
                cv=kfold,
                return_train_score=True,
                return_estimator=True,
            )
            # テストセットの分類精度で最も性能の良かった学習した分類器をリストとして保持しておく
            self.supervised_model.append(
                score["estimator"][np.argmax(score["test_score"])]
            )
            self.iris_supervised[model_name] = score["test_score"]
            for i in range(5):
                print(
                    "test score:{:.3f}, train score:{:.3f}".format(
                        score["test_score"][i], score["train_score"][i]
                    )
                )

    def get_supervised(self):
        return self.iris_supervised

    def best_supervised(self):
        iris_supervised_T = self.iris_supervised.describe().T
        best_score = iris_supervised_T["mean"].max()
        max_method = iris_supervised_T["mean"].idxmax()
        return max_method, best_score

    def plot_feature_importances_all(self):
        tree_model = [
            "DecisionTreeClassifier",
            "RandomForest",
            "GradientBoostingClassifier",
        ]
        # tree_modelに入っている学習済み分類器はself.supervised_modelの3,6,7番目の要素
        # supervised_modelがない場合はどうなるのでしょう？モデルの順番が入れ替わってしまったら？この関数は動くのでしょうか
        for model, model_name in zip(
            [
                self.supervised_model[3],
                self.supervised_model[6],
                self.supervised_model[7],
            ],
            tree_model,
        ):
            plt.figure()
            plt.barh(
                range(len(self.iris.feature_names)),
                model.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(len(self.iris.feature_names)), self.iris.feature_names)
            plt.xlabel("Feature importance:{}".format(model_name))

    def visualize_decision_tree(self):
        dot_data = export_graphviz(
            self.supervised_model[3],
            out_file=None,
            feature_names=self.iris.feature_names,
            class_names=self.iris.target_names,
            filled=True,
            impurity=False,
        )
        return graphviz.Source(dot_data)
