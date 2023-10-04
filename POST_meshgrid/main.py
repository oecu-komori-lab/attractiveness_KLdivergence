import numpy as np
import postprocess_multi_dim as post

def main(session, name_list, size):
    # Create models by passing compared points and preference data
    post.model_create(name_list, session)

    # Predict the distribution for each model
    post.predict_dist(name_list, session=session, length=1000, size=size)

    # Synthesize the main results based on the models
    post.main_synthesize(name_list, session)

if __name__ == '__main__':
    SESSION = ["cute", "beauty"]
    SIZE = 214359
    name_list = np.loadtxt("../data/name_list.csv", dtype="unicode")

    # Generate the point mesh
    post.point_mesh()

    # Iterate through sessions and perform the main processing
    for session in SESSION:
        main(session=session, name_list=name_list, size=SIZE)

