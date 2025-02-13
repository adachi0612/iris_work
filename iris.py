import itertools
from typing import Any, Union

# 可視化ライブラリを読み込む
import graphviz
import inflect
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Source
from seaborn import PairGrid
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans

# 教師あり学習の各手法を読み込む
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
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
        正解ラベルや正解種名を追加したDataFrameも作成する。
        """
        self.iris = load_iris()
        # FIXME: dataframe はdfと略すことが多いのそちらにしてださい。
        # TODO: irisという名前も抽象度が低いので、他の名前の方がいいですが..., お任せします
        self.iris_df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        # NOTE: 正解ラベル付きのDataFrameを作成
        self.iris_df_label = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        self.iris_df_label["Label"] = self.iris.target
        # NOTE: 正解種名付きのDataFrameを作成
        self.iris_df_species = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        self.iris_df_species["Species"] = np.array(
            self.iris.target_names[i] for i in self.iris.target
        )
        # DataFrameから特徴量と、正解ラベルの情報を格納する またよく使うものをより簡潔な名前で呼び出せるようにする
        self.data, self.target, self.feature_names, self.target_names = (
            self.iris_df.loc[:, self.iris.feature_names],
            self.iris_df_label["Label"],
            self.iris.feature_names,
            self.iris.target_names,
        )

    def get(self) -> pd.DataFrame:
        """正解ラベル付きのirisのデータセットのDataFrameを出力する。

        Returns:
            pd.DataFrame: 正解ラベル付きのirisのデータセットのDataFrame
        """
        return self.iris_df_label

    def get_correlation(self) -> pd.DataFrame:
        """各特徴量ごとの相関係数を求める。

        Returns:
            pd.DataFrame: 相関係数のDataFrame
        """
        return self.iris_df.corr()

    def pair_plot(self, diag_kind: str = "hist") -> PairGrid:
        """seabornライブラリのpairplotメソッドを用いて、各特徴量間の散布図や、ヒストグラムを表示。

        Args:
            diag_kind (str, optional): 対角成分のグラフの種類の指定 Defaults to "hist".

        Returns:
            PairGrid: 特徴量間ペアプロット
        """
        # FIXME: iris_dataframe_speciesは他の関数で使う予定がないなら、selfは使わなくてもいいです。
        return sns.pairplot(
            self.iris_df_species,
            hue="Species",
            diag_kind=diag_kind,
            markers="o",
            palette="tab10",
        )

    def all_supervised(self, n_neighbors: int = 4, n_splits: int = 5) -> None:
        """教師あり学習の主要な分類器にirisの特徴量を学習させた場合の訓練データ、テストデータに対する性能比較

        Args:
            n_neighbors (int, optional): k近傍法の、近傍点パラメータ. Defaults to 4.
            n_splits (int, optional): k分割交差検証の分割数. Defaults to 5.
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
        # NOTE: k(=n_splits)分割の交差検証を行う
        kfold = KFold(n_splits=n_splits)
        self.iris_supervised = pd.DataFrame()
        self.supervised_model = []

        for model_name, model in self.model_collections.items():
            print("=== {} ===".format(model_name))
            score = cross_validate(
                model,
                self.data,
                self.target,
                cv=kfold,
                return_train_score=True,
                return_estimator=True,
            )
            # NOTE: テストセットの分類精度で最も性能の良かった学習した分類器をリストとして保持しておく
            self.supervised_model.append(
                score["estimator"][np.argmax(score["test_score"])]
            )
            self.iris_supervised[model_name] = score["test_score"]
            for i in range(n_splits):
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
            tuple[str | float, int]: 性能が最大の分類器の名前とその性能
        """
        iris_supervised_T = self.iris_supervised.describe().T
        best_score = iris_supervised_T["mean"].max()
        max_method = iris_supervised_T["mean"].idxmax()
        return max_method, best_score

    def plot_feature_importances_all(self, n_splits: int = 5) -> None:
        """上で学習したものを持ってくるのではなく、この関数だけで、決定木、ランダムフォレスト、勾配ブースティング回帰木のfeature_importancesを可視化する
        同様にk(=n_splits)分割交差検証を行う。


        Args:
            n_splits (int, optional): k分割交差検証の分割数. Defaults to 5.
        """
        # supervised_modelがない場合はどうなるのでしょう？モデルの順番が入れ替わってしまったら？この関数は動くのでしょうか

        tree_models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
        }
        for i, (model_name, model) in enumerate(tree_models.items()):
            # NOTE: k(=n_splits)分割の交差検証を行う
            kfold = KFold(n_splits=n_splits)
            tree_score = cross_validate(
                model, self.data, self.target, cv=kfold, return_estimator=True
            )
            # NOTE: 交差検証の訓練データを学習させたモデルを抽出
            model = tree_score["estimator"][i]
            plt.figure()
            plt.barh(
                range(len(self.feature_names)),
                model.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(len(self.feature_names)), self.feature_names)
            plt.xlabel("Feature importance:{}".format(model_name))

    def visualize_decision_tree(self) -> Source:
        """決定木を可視化する

        Returns:
            Source: graphvizを利用して、決定木のクラス分類過程を図示する。
        """
        dot_data = export_graphviz(
            DecisionTreeClassifier().fit(self.data, self.target),
            out_file=None,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            impurity=False,
        )
        return graphviz.Source(dot_data)

    def plot_scaled_data(
        self,
        n_splits: int = 5,
        scalers: tuple[str | BaseEstimator, ...] = (
            "Original",
            MinMaxScaler(),
            StandardScaler(),
            RobustScaler(),
            Normalizer(),
        ),
    ) -> None:
        """データ前処理(初期値は変換なし, MinMaxScaler, StandardScaler, RobustScaler, Normalizerの5種類)を行い、
        k分割交差検証（初期値はk=5）で分類器の性能を測る

        Args:
            n_splits (int, optional): k分割交差検証の分割数. Defaults to 5.
            scalers (tuple[str  |  BaseEstimator, ...], optional): スケール変換の種類を指定. Defaults to ( "Original", MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), ).
        """
        # NOTE: k(=n_splits)分割の交差検証を行う
        kfold = KFold(n_splits=n_splits)

        linear_svc = LinearSVC()
        # NOTE: 特徴量の名前の組み合わせ（irisのデータセットの場合、4C2で6通り）を作成
        feature_names_combination_list = list(
            itertools.combinations(range(len(self.feature_names)), 2)
        )
        feature_names_combination_len = len(feature_names_combination_list)
        # NOTE: グラフの間隔を調整するためのパラメーターの設定
        wspace, hspace = 0.4, 0.3

        # NOTE: "Original"がscalersのタプルに入っている場合
        if "Original" in scalers and scalers != ("Original"):
            for i, (train_index, test_index) in enumerate(
                kfold.split(self.data, self.target)
            ):

                fig, axes = plt.subplots(
                    feature_names_combination_len, len(scalers), figsize=(12, 20)
                )
                # NOTE: グラフの間隔を調整する
                fig.tight_layout()
                fig.subplots_adjust(wspace=wspace, hspace=hspace)
                scores = cross_validate(
                    linear_svc,
                    self.data,
                    self.target,
                    cv=kfold,
                    return_train_score=True,
                )
                print(
                    "{: <16}: test score: {:.2f} train score: {:.2f}".format(
                        "Original", scores["test_score"][i], scores["train_score"][i]
                    )
                )
                # NOTE: "Original"データのスケール変換後の特徴量をプロット
                for j, (Feature_0, Feature_1) in enumerate(
                    feature_names_combination_list
                ):
                    # NOTE: グラフの位置を求める
                    ax = axes[j, 0]
                    # NOTE: 上で求めた異なる2つを取り出す組み合わせの番号と特徴量の名前を対応させる
                    Feature_0_name = self.feature_names[Feature_0]
                    Feature_1_name = self.feature_names[Feature_1]
                    # NOTE: 訓練データをプロットする
                    ax.scatter(
                        self.iris_df.loc[train_index, Feature_0_name],
                        self.iris_df.loc[train_index, Feature_1_name],
                        marker="o",
                    )
                    # NOTE:テストデータをプロットする
                    ax.scatter(
                        self.iris_df.loc[test_index, Feature_0_name],
                        self.iris_df.loc[test_index, Feature_1_name],
                        marker="^",
                    )
                    ax.set_xlabel(self.feature_names[Feature_0])
                    ax.set_ylabel(self.feature_names[Feature_1])
                    ax.set_title("Original")
                # NOTE: 各前処理した後のデータのスケール変換後の特徴量をプロット
                for scaler in scalers[1:]:
                    pipe = make_pipeline(scaler, linear_svc)
                    pipe.fit(self.data, self.target)
                    scores = cross_validate(
                        pipe, self.data, self.target, cv=kfold, return_train_score=True
                    )
                    # NOTE: スケール変換した特徴量を求める
                    scaler_data = pipe.named_steps[
                        scaler.__class__.__name__.lower()
                    ].transform(self.data)
                    print(
                        "{: <16}: test score: {:.2f} train score: {:.2f}".format(
                            scaler.__class__.__name__,
                            scores["test_score"][i],
                            scores["train_score"][i],
                        )
                    )
                    for k, (Feature_0, Feature_1) in enumerate(
                        feature_names_combination_list
                    ):
                        # NOTE: グラフの位置を求める
                        ax = axes[k, scalers.index(scaler)]
                        # NOTE: 訓練データをプロットする
                        ax.scatter(
                            scaler_data[train_index, Feature_0],
                            scaler_data[train_index, Feature_1],
                            marker="o",
                        )
                        # NOTE:テストデータをプロットする
                        ax.scatter(
                            scaler_data[test_index, Feature_0],
                            scaler_data[test_index, Feature_1],
                            marker="^",
                        )
                        ax.set_xlabel(self.feature_names[Feature_0])
                        ax.set_ylabel(self.feature_names[Feature_1])
                        ax.set_title(scaler.__class__.__name__)
                plt.show()
        # NOTE: "Original"のみがscalersのタプルに入っている場合
        if scalers == ("Original"):
            print("スケール変換器が必要です.")
        # NOTE: "Original"がscalersのタプルに入っていない場合
        elif "Original" not in scalers:
            for i, (train_index, test_index) in enumerate(
                kfold.split(self.data, self.target)
            ):

                fig, axes = plt.subplots(
                    feature_names_combination_len, len(scalers), figsize=(12, 20)
                )
                # NOTE: グラフの間隔を調整する
                fig.tight_layout()
                fig.subplots_adjust(wspace=wspace, hspace=hspace)
                # NOTE: 各前処理した後のデータのスケール変換後の特徴量をプロット
                for scaler in scalers:
                    pipe = make_pipeline(scaler, linear_svc)
                    pipe.fit(self.data, self.target)
                    scores = cross_validate(
                        pipe, self.data, self.target, cv=kfold, return_train_score=True
                    )
                    # NOTE: スケール変換した特徴量を求める
                    scaler_data = pipe.named_steps[
                        scaler.__class__.__name__.lower()
                    ].transform(self.data)
                    print(
                        "{: <16}: test score: {:.2f} train score: {:.2f}".format(
                            scaler.__class__.__name__,
                            scores["test_score"][i],
                            scores["train_score"][i],
                        )
                    )
                    for k, (Feature_0, Feature_1) in enumerate(
                        feature_names_combination_list
                    ):
                        # NOTE: グラフの位置を求める
                        ax = axes[k, scalers.index(scaler)]
                        # NOTE: 訓練データをプロットする
                        ax.scatter(
                            scaler_data[train_index, Feature_0],
                            scaler_data[train_index, Feature_1],
                            marker="o",
                        )
                        # NOTE:テストデータをプロットする
                        ax.scatter(
                            scaler_data[test_index, Feature_0],
                            scaler_data[test_index, Feature_1],
                            marker="^",
                        )
                        ax.set_xlabel(self.feature_names[Feature_0])
                        ax.set_ylabel(self.feature_names[Feature_1])
                        ax.set_title(scaler.__class__.__name__)
                plt.show()

    def plot_pca(self, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """データセットの特徴量の、第k主成分(k=n_components)までを求め、第二主成分までを2次元散布図に描き、
        第k主成分までをのヒートマップを作成

        Args:
            n_components (int, optional): PCAの主成分の数を決める. Defaults to 2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, PCA]: 標準化したデータと、そのデータをさらに主成分分析で次元削減したデータとその主成分
        """
        # NOTE: 主成分分析で特徴量データの次元削減
        # NOTE: データを標準化する
        X_scaled = StandardScaler().fit_transform(self.iris_df)
        # NOTE: pd.DataFrame形式に変換する
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        X_pca = PCA(n_components=n_components).fit(X_scaled)
        df_pca = pd.DataFrame(
            X_pca.transform(X_scaled),
            columns=["Feature " + str(i) for i in range(n_components)],
        )
        # NOTE: 主成分分析で次元削減をしたデータの2次元散布図を作成
        if n_components > 1:
            mglearn.discrete_scatter(
                df_pca.iloc[:, 0], df_pca.iloc[:, 1], self.iris_df_label["Label"]
            )
            plt.legend(self.target_names, loc="best")
            plt.xlabel("First component")
            plt.ylabel("Second component")

        # NOTE: 主成分分析の、第一主成分、第二主成分...をヒートマップで表示
        # NOTE: 自然数の列1,2,3... を 1st 2nd 3rd...と変換するinflectライブラリを使用する
        p = inflect.engine()
        plt.matshow(X_pca.components_)
        plt.yticks(
            range(n_components),
            [p.ordinal(i + 1) + " component" for i in range(n_components)],
        )
        plt.xticks(range(len(self.feature_names)), self.feature_names)
        plt.xlabel("Feature")
        plt.ylabel("Principal components")
        plt.colorbar()

        return X_scaled, df_pca, X_pca

    def plot_nmf(self, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, NMF]:
        """データセットの特徴量をNMF(非負値行列因子分解)で変換した成分k(k=n_components)までを求め、成分1と2を2次元散布図に描き、
        成分kまでをのヒートマップを作成

        Args:
            n_components (int, optional): NMFの求める成分数. Defaults to 2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, NMF]: 標準化したデータと、そのデータをさらにNMFで次元削減したデータとその成分
        """
        # NOTE: NMFで特徴量データの次元削減
        # NOTE: NMFは非負データしか扱えないのでMinMaxScalerでデータを変換する
        # X_scaled = MinMaxScaler().fit_transform(self.iris_df)
        # NOTE: pd.DataFrame形式に変換する
        X_scaled = pd.DataFrame(self.iris_df, columns=self.feature_names)
        X_nmf = NMF(n_components=n_components).fit(X_scaled)
        df_nmf = pd.DataFrame(
            X_nmf.transform(X_scaled),
            columns=["Component " + str(i) for i in range(n_components)],
        )
        # NOTE: NMFで次元削減をしたデータの2次元散布図を作成
        if n_components > 1:
            mglearn.discrete_scatter(
                df_nmf.iloc[:, 0], df_nmf.iloc[:, 1], self.iris_df_label["Label"]
            )
            plt.legend(self.target_names, loc="best")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        # NOTE: NMFの、第一主成分、第二主成分...をヒートマップで表示
        plt.matshow(X_nmf.components_)
        plt.yticks(range(n_components), list(df_nmf.columns))
        plt.xticks(range(len(self.feature_names)), self.feature_names)
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.colorbar()

        return X_scaled, df_nmf, X_nmf

    def plot_tsne(self) -> None:
        """t-SNEを用いた教師なし学習での分類の2次元散布図での可視化"""
        tsne = TSNE(random_state=0)
        X_tsne = tsne.fit_transform(self.data)
        plt.xlim(X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1)
        plt.ylim(X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1)
        for i in range(len(X_tsne[:, 0])):
            plt.text(
                X_tsne[i, 0],
                X_tsne[i, 1],
                str(self.iris_df_label["Label"][i]),
                fontdict={"weight": "bold", "size": 9},
            )
        plt.xlabel("t-SNE feature0")
        plt.ylabel("t-SNE feature1")

    def plot_k_means(self, n_clusters: int = 3) -> None:
        """KMeansでのクラスタリング結果の可視化と、クラスタセンタのプロット、および実際のクラス分類との比較

        Args:
            n_clusters (int, optional): KMeansのクラスタ数. Defaults to 3.
        """
        # NOTE: KMeansでクラスタリングした結果を2次元散布図で表すためにPCAで次元を2まで削減する
        X_pca = PCA(n_components=2).fit_transform(self.iris_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(X_pca)
        y_pca = kmeans.predict(X_pca)
        # NOTE: KMeansで予測したクラスタリング結果を表示
        print("KMeans法で予測したラベル:\n{}".format(y_pca))
        markers = ["^", "v", "o"]
        colors = ["b", "g", "r"]
        cluster_center_size = 20
        plt.figure()
        for i in np.unique(y_pca):
            plt.scatter(
                X_pca[y_pca == i, 0],
                X_pca[y_pca == i, 1],
                c=colors[i],
                marker=markers[i],
            )
        # NOTE: 各クラスタの重心をプロット
        plt.plot(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="black",
            marker="*",
            markersize=cluster_center_size,
            linestyle=" ",
        )

        # NOTE: 元のデータのクラス分類
        print("実際のラベル:\n{}".format(self.target.to_numpy()))
        plt.figure()
        for i in np.unique(self.target):
            plt.scatter(
                X_pca[self.target == i, 0],
                X_pca[self.target == i, 1],
                c=colors[i],
                marker=markers[i],
            )
        plt.plot(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="black",
            marker="*",
            markersize=cluster_center_size,
            linestyle=" ",
        )
