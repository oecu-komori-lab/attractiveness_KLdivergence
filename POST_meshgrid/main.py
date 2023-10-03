import numpy as np
import postprocess_multi_dim as post


def main(session, name_list, size):
    post.model_create(name_list, session)  # 比較した点と選好結果のデータを投げてモデル作成
    post.predict_dist(name_list, session=session, length=1000, size=size)
    post.main_synthesize(name_list, session)


if __name__ == '__main__':
    SESSION = ["cute","beauty"]
    SIZE = 214359
    name_list = np.loadtxt("../data/name_list.csv", dtype = "unicode")
    post.point_mesh()

    for session in SESSION:
        main(session=session, name_list=name_list, size=SIZE)
