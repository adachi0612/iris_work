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

    def get(self):
        # データフレームを作成
        iris_data = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        # ラベルを追加
        iris_data["Label"] = self.iris.target
        return iris_data
