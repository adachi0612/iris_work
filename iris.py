import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


class AnalyzeIris:
    def __init__(self):
        # load_iris()でIrisデータセットを読み込む
        self.iris = load_iris()
        # 正解ラベルのない各特徴量のデータフレームを作成
        self.iris_data = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)

    def get(self):
        self.iris_data_label = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        # ラベルの列を追加
        self.iris_data_label["Label"] = self.iris.target
        return self.iris_data_label

    def get_correlation(self):
        # 各特徴量間の相関係数を求める
        return self.iris_data.corr()

    def pair_plot(self, diag_kind="hist"):
        # 正解ラベルのない各特徴量のデータフレームを作成
        self.iris_data_species = pd.DataFrame(
            self.iris.data, columns=self.iris.feature_names
        )
        # 種の正解ラベル列を追加
        self.iris_data_species["Species"] = np.array(
            self.iris.target_names[i] for i in self.iris.target
        )
        # seabornでペアプロットを行う
        return sns.pairplot(
            self.iris_data_species,
            hue="Species",
            diag_kind=diag_kind,
            markers="o",
            palette="tab10",
        )

    # def all_supervised(self, n_neighbors=4):
