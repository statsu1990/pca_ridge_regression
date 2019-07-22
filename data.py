import numpy as np


class DataMultico:
    """description of class"""
    def __init__(self, sample_num, 
                 multico_x_nums_inSet, error_stds_inSet, 
                 coefs_inSet, bias, error_std_y):
        # サンプル数
        self.SAMPLE_NUM = sample_num
        
        # 多重共線形性のある説明変数セット
        # [3, 2, 4] -> x1_1, x1_2, x1_3, x2_1, x2_2, x3_1, x3_2, x3_3, x3_4
        self.MULTICO_X_NUMS_INSET = multico_x_nums_inSet

        # 各説明変数セットの誤差の標準偏差
        self.ERROR_STDS_INSET = error_stds_inSet

        # 各説明変数セットの偏回帰係数
        self.COEFS_INSET = coefs_inSet

        # バイアス
        self.BIAS = bias
        
        self.ERROR_STD_Y = error_std_y

        return

    def make_data(self):

        # initial x and y
        self.y = np.full((self.SAMPLE_NUM, 1), self.BIAS) + np.random.normal(0, self.ERROR_STD_Y, (self.SAMPLE_NUM, 1))
        self.x = None

        # x and y
        for multico_x_num_inSet, error_std_inSet, coef_inSet in zip(self.MULTICO_X_NUMS_INSET, self.ERROR_STDS_INSET, self.COEFS_INSET):
            # x in set
            _x = np.random.normal(0, 1, (self.SAMPLE_NUM, 1)) + np.random.normal(0, error_std_inSet, (self.SAMPLE_NUM, multico_x_num_inSet))
            self.x = np.concatenate([self.x, _x], axis=1) if self.x is not None else _x
            # +y
            self.y = self.y + np.sum(_x, axis=1)[:, np.newaxis] * coef_inSet

        # ref coef
        self.REF_COEF = []
        for multico_x_num_inSet, coef_inSet in zip(self.MULTICO_X_NUMS_INSET, self.COEFS_INSET):
            self.REF_COEF.extend([coef_inSet] * multico_x_num_inSet)

        return

