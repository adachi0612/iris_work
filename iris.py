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
from randomcolor import RandomColor
from scipy.cluster.hierarchy import dendrogram, linkage
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
        iris_df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
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
        self.df, self.target, self.feature_names, self.target_names = (
            iris_df,
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

    def get_with_species(self) -> pd.DataFrame:
        """正解種名付きのirisのデータセットのDataFrameを出力する。

        Returns:
            pd.DataFrame: 正解種名付きのirisのデータセットのDataFrame
        """
        return self.iris_df_species

    def get_correlation(self) -> pd.DataFrame:
        """各特徴量ごとの相関係数を求める。

        Returns:
            pd.DataFrame: 相関係数のDataFrame
        """
        return self.df.corr()

    def pair_plot(self, diag_kind: str = "hist") -> PairGrid:
        """seabornライブラリのpairplotメソッドを用いて、各特徴量間の散布図や、ヒストグラムを表示。

        Args:
            diag_kind (str, optional): 対角成分のグラフの種類の指定 Defaults to "hist".

        Returns:
            PairGrid: 特徴量間ペアプロット
        """
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
        # NOTE: k(=n_splits)分割の交差検証を行う
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.iris_supervised = pd.DataFrame()
        self.supervised_model = []

        for model_name, model in self.model_collections.items():
            print("=== {} ===".format(model_name))
            score = cross_validate(
                model,
                self.df,
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

        tree_models = {
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
        }
        for i, (model_name, model) in enumerate(tree_models.items()):
            # NOTE: k(=n_splits)分割の交差検証を行う
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            tree_score: dict[str, Any] = cross_validate(
                model, self.df, self.target, cv=kfold, return_estimator=True
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
            DecisionTreeClassifier().fit(self.df, self.target),
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
    ) -> list[list[float]]:
        """データ前処理(初期値は変換なし, MinMaxScaler, StandardScaler, RobustScaler, Normalizerの5種類)を行い、
        k分割交差検証（初期値はk=5）で分類器の性能を測る

        Args:
            n_splits (int, optional): k分割交差検証の分割数. Defaults to 5.
            scalers (tuple[str  |  BaseEstimator, ...], optional): スケール変換の種類を指定. Defaults to ( "Original", MinMaxScaler(), StandardScaler(), RobustScaler(), Normalizer(), ).

        Returns:
            list[list[float]]: 変換後の交差検証の各回の訓練データ
        """
        train_data = []
        # NOTE: k(=n_splits)分割の交差検証を行う
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        linear_svc = LinearSVC()
        # NOTE: 特徴量の名前の組み合わせ（irisのデータセットの場合、4C2で6通り）を作成
        list_feature_combination = list(
            itertools.combinations(range(len(self.feature_names)), 2)
        )
        len_feature_combination = len(list_feature_combination)
        wspace, hspace = 0.3, 0.5
        # NOTE: データ数や特徴量の数は同じなので、分割の仕方はデータ変換前後でもかわらずこのように書いても良い”
        for i, (train_index, test_index) in enumerate(
            kfold.split(self.df, self.target)
        ):
            fig, axes = plt.subplots(
                len_feature_combination, len(scalers), figsize=(12, 20)
            )
            fig.tight_layout()
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
            for scaler in scalers:
                # NOTE:
                X_scaled, y = self.df.values, self.target
                # NOTE: scalerが"Original"以外の場合で、データのスケール変換を行う
                if scaler != "Original":
                    X_scaled = scaler.fit_transform(X_scaled)
                scores = cross_validate(
                    linear_svc, X_scaled, y, cv=kfold, return_train_score=True
                )
                print(
                    "{: <16}: test score: {:.2f} train score: {:.2f}".format(
                        str(scaler),
                        scores["test_score"][i],
                        scores["train_score"][i],
                    )
                )
                for k, (Feature_0, Feature_1) in enumerate(list_feature_combination):
                    ax = axes[k, scalers.index(scaler)]
                    # NOTE: 訓練データをプロットする
                    ax.scatter(
                        X_scaled[train_index, Feature_0],
                        X_scaled[train_index, Feature_1],
                        marker="o",
                    )
                    # NOTE:テストデータをプロットする
                    ax.scatter(
                        X_scaled[test_index, Feature_0],
                        X_scaled[test_index, Feature_1],
                        marker="^",
                    )
                    ax.set_xlabel(self.feature_names[Feature_0])
                    ax.set_ylabel(self.feature_names[Feature_1])
                    ax.set_title(str(scaler))
            plt.show()
            train_data.append(X_scaled[train_index])
        return train_data

    def plot_decomposition(
        self, data: pd.DataFrame, Decomposition: BaseEstimator
    ) -> tuple[pd.DataFrame, BaseEstimator]:
        """PCAやNMFといった次元削減を行った後に、２次元散布図にデータをプロットする

        Args:
            data (pd.DataFrame): 次元削減と可視化をするデータ
            Decomposition (BaseEstimator): PCAやNMFなど

        Returns:
            tuple[pd.DataFrame, BaseEstimator]: 次元削減したデータを格納したデータフレームと、学習済みのモデルを出力、n_componentsが1以下の場合はエラー文を表示
        """
        X_decomposition = Decomposition.fit(data)
        df_decomposition = pd.DataFrame(
            X_decomposition.transform(data),
            columns=[
                str(i + 1) + " " + "Component"
                for i in range(Decomposition.n_components)
            ],
        )

        if Decomposition.n_components > 1:
            mglearn.discrete_scatter(
                df_decomposition.iloc[:, 0], df_decomposition.iloc[:, 1], self.target
            )
            plt.legend(self.target_names, loc="best")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")

            plt.matshow(X_decomposition.components_)
            plt.yticks(
                range(Decomposition.n_components), list(df_decomposition.columns)
            )
            plt.xticks(range(len(self.feature_names)), self.feature_names)
            plt.xlabel("Feature")
            plt.ylabel(Decomposition.__class__.__name__ + " " + "components")
            plt.colorbar()
            return df_decomposition, X_decomposition

        else:
            print("散布図を作成するために、n_componentsには2以上の値を入力してください")
            return df_decomposition, X_decomposition

    def plot_pca(
        self, n_components: int = 2
    ) -> tuple[pd.DataFrame, pd.DataFrame, BaseEstimator]:
        """データセットの特徴量の、第k主成分(k=n_components)までを求め、第二主成分までを2次元散布図に描き、
        第k主成分までをのヒートマップを作成

        Args:
            n_components (int, optional): PCAの主成分の数を決める. Defaults to 2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, BaseEstimator]: 標準化したデータと、そのデータをさらに主成分分析で次元削減したデータとその主成分
        """
        # NOTE: 主成分分析で特徴量データの次元削減
        # NOTE: データを標準化する
        X_scaled = StandardScaler().fit_transform(self.df)
        # NOTE: pd.DataFrame形式に変換する
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        df_pca, X_pca = self.plot_decomposition(
            data=X_scaled, Decomposition=PCA(n_components=n_components)
        )

        return X_scaled, df_pca, X_pca

    def plot_nmf(
        self, n_components: int = 2
    ) -> tuple[pd.DataFrame, pd.DataFrame, BaseEstimator]:
        """データセットの特徴量をNMF(非負値行列因子分解)で変換した成分k(k=n_components)までを求め、成分1と2を2次元散布図に描き、
        成分kまでをのヒートマップを作成

        Args:
            n_components (int, optional): NMFの求める成分数. Defaults to 2.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, BaseEstimator]: 標準化したデータと、そのデータをさらにNMFで次元削減したデータとその成分
        """
        # NOTE: NMFで特徴量データの次元削減
        # NOTE: NMFは非負データしか扱えないのでMinMaxScalerでデータを変換する
        # X_scaled = MinMaxScaler().fit_transform(self.data)
        # NOTE: pd.DataFrame形式に変換する
        X_scaled = pd.DataFrame(self.df, columns=self.feature_names)
        df_nmf, X_nmf = self.plot_decomposition(
            data=X_scaled, Decomposition=NMF(n_components=n_components)
        )

        return X_scaled, df_nmf, X_nmf

    def plot_tsne(self) -> None:
        """t-SNEを用いた教師なし学習での分類の2次元散布図での可視化"""
        tsne = TSNE(random_state=0)
        X_tsne = tsne.fit_transform(self.df)
        plt.xlim(X_tsne[:, 0].min() - 1, X_tsne[:, 0].max() + 1)
        plt.ylim(X_tsne[:, 1].min() - 1, X_tsne[:, 1].max() + 1)
        for i in range(len(X_tsne[:, 0])):
            plt.text(
                X_tsne[i, 0],
                X_tsne[i, 1],
                str(self.target[i]),
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
        X_pca = PCA(n_components=2).fit_transform(self.df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(X_pca)
        y_pca = kmeans.predict(X_pca)
        # NOTE: KMeansで予測したクラスタリング結果を表示
        print("KMeans法で予測したラベル:\n{}".format(y_pca))
        cluster_center_size = 20
        plt.figure()
        mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y_pca)
        # NOTE: 各クラスタの重心をプロット
        plt.plot(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="black",
            marker="*",
            markersize=cluster_center_size,
            linestyle=" ",
        )
        plt.show()

        # NOTE: 元のデータのクラス分類
        print("実際のラベル:\n{}".format(self.target.to_numpy()))
        plt.figure()
        mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], self.target.to_numpy())
        plt.plot(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c="black",
            marker="*",
            markersize=cluster_center_size,
            linestyle=" ",
        )
        plt.show()

    def plot_dendrogram(
        self,
        truncate: bool = False,
        mode: str = "lastp",
        p: int = 10,
        method: str = "ward",
        metric: str = "euclidean",
    ) -> None:
        """凝集型クラスタリングの結果をデンドログラムで表示する

        Args:
            truncate (bool, optional): Trueでデンドログラムの一部だけを見る . Defaults to False.
            mode (str, optional): デンドログラムのどの部分までを見るかのメソッド. Defaults to "lastp".
            p (int, optional): mode="lastp"を選んだときのパラメータ. Defaults to 10.
            method (str, optional): 連結度を調べる際、どの指標を用いるか. Defaults to "ward".
            metric (str, optional): どの距離尺度を採用するか. Defaults to "euclidean".
        """
        # NOTE: 凝集型クラスタリングをwardで行った際のブリッジ距離を示す配列を求める
        linkage_array = linkage(self.df, method=method, metric=metric)
        if truncate:
            dendrogram(linkage_array, truncate_mode=mode, p=p)
            plt.show()
        else:
            dendrogram(linkage_array)
            plt.show()

    def plot_dbscan(
        self,
        scaling: bool = True,
        eps: float = 0.5,
        min_samples: int = 5,
        pairs: list[tuple[int, int]] = [(2, 3)],
    ) -> None:
        """DBSCANのクラスタリング結果を、データの２つの特徴量を縦軸と横軸に取った図で可視化する。

        Args:
            scaling (bool, optional): Trueで特徴量を標準化する. Defaults to True.
            eps (float, optional): DBSCANのパラメータ. Defaults to 0.5.
            min_samples (int, optional): DBSCANのパラメータ. Defaults to 5.
            pairs (list[tuple[int, int]], optional): どの特徴量同士を2次元散布図に表示するかを入力. Defaults to [(2, 3)].
        """
        if scaling:
            data_scaled = StandardScaler().fit_transform(self.df)
        else:
            data_scaled = self.df
        dbscan = DBSCAN(min_samples=min_samples, eps=eps)
        # NOTE:ノイズのクラス分類が
        clusters = dbscan.fit_predict(data_scaled)

        for pair in pairs:
            mglearn.discrete_scatter(
                data_scaled[:, pair[0]], data_scaled[:, pair[1]], clusters
            )
            plt.xlabel("Feature " + str(pair[0]))
            plt.ylabel("Feature " + str(pair[1]))
            plt.show()
        print("Cluster memberships:\n{}".format(clusters))
