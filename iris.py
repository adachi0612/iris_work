# TypeHintsのためのライブラリを読み込む
from typing import Any, Union

# 可視化ライブラリを読み込む
import graphviz
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Source
from seaborn import PairGrid

# 教師あり学習の各手法を読み込む
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import Bunch

"""
# TODO: 
# 全体的に、コメントはgoogle Docs形式にして、type hintを追加してください。
# 課題に入っているREADMEに詳細は書かれています。
# また、保守や拡張性を考えて実装することを心がけてください。kfold = KFold(n_splits=5)のところなど、数字をベタ書きするのはお勧めしません。

"""


class AnalyzeIris:
    """"""

    def __init__(self) -> None:
        """クラスの初期化を行い、irisのデータセットを読み込みDataFrame形式にして出力する。

        Returns:
            None: Return value
        """
        self.iris = load_iris()
        # FIXME: dataframe はdfと略すことが多いのそちらにしてださい。
        # TODO: irisという名前も抽象度が低いので、他の名前の方がいいですが..., お任せします
        self.iris_df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)

    def get(self) -> pd.DataFrame:
        """正解ラベル付きのirisのデータセットのDataFrameを出力する。

        Returns:
            pd.DataFrame: 正解ラベル付きのirisのデータセットのDataFrame
        """
        self.iris_df_label = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        self.iris_df_label["Label"] = self.iris.target
        return self.iris_df_label

    def get_correlation(self) -> pd.DataFrame:
        """各特徴量ごとの相関係数を求める。

        Returns:
            pd.DataFrame: クラスの初期化をした際に作成したirisのデータセットのDataFrameを用いて、
            相関係数を求め、DataFrameとして出力する。
        """
        return self.iris_df.corr()

    def pair_plot(self, diag_kind: str = "hist") -> PairGrid:
        """seabornライブラリのpairplotメソッドを用いて、各特徴量間の散布図や、ヒストグラムを表示。

        Args:
            diag_kind(str): pairplotの対角成分の図の種類を指定

        Returns:
            PairGrid: 各特徴量間のpairplotを表示
        """
        # FIXME: iris_dataframe_speciesは他の関数で使う予定がないなら、selfは使わなくてもいいです。
        iris_df_species = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        iris_df_species["Species"] = np.array(
            self.iris.target_names[i] for i in self.iris.target
        )
        return sns.pairplot(
            iris_df_species,
            hue="Species",
            diag_kind=diag_kind,
            markers="o",
            palette="tab10",
        )

    def all_supervised(self, n_neighbors: int = 4, n_splits: int = 5) -> None:
        """教師あり学習の主要な分類器にirisの特徴量を学習させた場合の訓練データ、テストデータに対する性能比較

        Args:
            n_neighbors(int): k最近傍法での近傍オブジェクト数を指定（初期値:4）
            n_splits(int): k分割交差検証での、分割数を指定（初期値:5）

        Returns:
            None: for文中のprint関数で次々に各分類器の性能を表示
        """
        # FIXME: 二つのリストにせずに、dictにした方がいいと思います。
        self.model_collections = {
            "LogisticRegression": LogisticRegression(),
            "LinearSVC": LinearSVC(),
            "SVC": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=n_neighbors),
            "LinearRegression": LinearRegression(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "MLPClassifier": MLPClassifier(),
        }
        # FIXME: 5という数字を変数にして、変更しやすくするといいです。下のプリント文もこの数字に対応させてください。
        # k(=n_splits)分割の交差検証を行う
        kfold = KFold(n_splits=n_splits)
        self.iris_supervised = pd.DataFrame()
        self.supervised_model = []

        for model_name, model in self.model_collections.items():
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

    def get_supervised(self) -> pd.DataFrame:
        """all_supervisedで行った分類器の性能の検証結果をDataFrame形式で表示する。

        Returns:
            pd.DataFrame: 分割数に応じた交差検証の結果を各分類器を列として表示する。
        """
        return self.iris_supervised

    def best_supervised(self) -> tuple[str | float, int]:
        """分類器の中で、交差検証の結果の平均値が最も高かった分離器とその平均値を出力する

        Returns:
            best_method(str): 分類器の名前の型は全て文字列のため、str型で出力

            best_score(float, int): 交差検証の結果は数値のため、int型かfloat型で出力
        """
        iris_supervised_T = self.iris_supervised.describe().T
        best_score = iris_supervised_T["mean"].max()
        max_method = iris_supervised_T["mean"].idxmax()
        return max_method, best_score

    def plot_feature_importances_all(self, n_splits: int = 5) -> None:
        """上で学習したものを持ってくるのではなく、この関数だけで、決定木、ランダムフォレスト、勾配ブースティング回帰木のfeature_importancesを可視化する
        同様にk(=n_splits)分割交差検証を行う。

        Args:
            n_splits(int): k分割交差検証での、分割数を指定（初期値:5）

        Returns:
            None: 縦軸に特徴量をと理、横軸に各特徴量のfeature_importancesの値をとった、横向きの棒グラフをそれぞれの分類器の場合で可視化する。
        """
        # supervised_modelがない場合はどうなるのでしょう？モデルの順番が入れ替わってしまったら？この関数は動くのでしょうか

        tree_models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
        }
        for i, (model_name, model) in enumerate(tree_models.items()):
            # k(=n_splits)分割の交差検証を行う
            kfold = KFold(n_splits=n_splits)
            tree_score = cross_validate(
                model, self.iris.data, self.iris.target, cv=kfold, return_estimator=True
            )
            # 交差検証の訓練データを学習させたモデルを抽出
            model = tree_score["estimator"][i]
            plt.figure()
            plt.barh(
                range(len(self.iris.feature_names)),
                model.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(len(self.iris.feature_names)), self.iris.feature_names)
            plt.xlabel("Feature importance:{}".format(model_name))

    def visualize_decision_tree(self) -> Source:
        """決定木を可視化する

        Returns:
            Source: graphvizを利用して、決定木のクラス分類過程を図示する。
        """
        dot_data = export_graphviz(
            self.supervised_model[3],
            out_file=None,
            feature_names=self.iris.feature_names,
            class_names=self.iris.target_names,
            filled=True,
            impurity=False,
        )
        return graphviz.Source(dot_data)
