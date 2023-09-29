import numpy as np
import pandas as pd

import postprocess_multi_dim as post


def main(session, name_list, size):
    #post.model_create(name_list, session)  # 比較した点と選好結果のデータを投げてモデル作成
    # post.predict_dist(name_list, session=session, length=1000, size=size)
    post.main_synthesize(name_list, session)
    #post.cov_for_synthesis(name_list, session, length=1000, size=size)
    # post.cov_synthesize(name_list, session)  # 平均予測平均最大値となる点とその他の点の共分散の合成
    # post.difference_variance_synthesized(session)
    # post.difference_mean_synthesized(session)
    # post.synthesized_dif_dist_p_value(session)


if __name__ == '__main__':
    SESSION = ["cute","beauty"]
    SIZE = 214359
    name_list = np.loadtxt("../data/name_list.csv", dtype = "unicode")
    # post.point_mesh()

    for session in SESSION:
        main(session=session, name_list=name_list, size=SIZE)
