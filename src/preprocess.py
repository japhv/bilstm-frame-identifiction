import pandas as pd

from nltk.corpus import framenet as fn
from itertools import islice


def writeTrainTest(exemplars_per_frame=100):

    columns = ["exemplar", "frame_id"]
    train_df = pd.DataFrame(columns=columns)
    valid_df = pd.DataFrame(columns=columns)
    test_df = pd.DataFrame(columns=columns)

    frames = pd.read_csv("../misc/frames_used.csv")

    frames_dir = {int(row["f_id"]): idx for idx, row in frames.iterrows()}

    for f_id, idx in frames_dir.items():
        e_set = [{
            "exemplar": e.text,
            "frame_id" :f_id}
            for e in fn.exemplars(frame=f_id)]

        no_of_exemplars = len(e_set)

        print("Working on frame:", f_id, no_of_exemplars)

        if no_of_exemplars < 10:
            continue


        idx_60 = int(abs(no_of_exemplars * 0.6))
        idx_80 = int(abs(no_of_exemplars * 0.8))

        train_df = train_df.append(e_set[:idx_60])
        valid_df = valid_df.append(e_set[idx_60:idx_80])
        test_df = test_df.append(e_set[idx_80:])

        train_df["label"] = train_df["frame_id"].apply(lambda x: frames_dir[int(x)])
        test_df["label"] = test_df["frame_id"].apply(lambda x: frames_dir[int(x)])
        valid_df["label"] = valid_df["frame_id"].apply(lambda x: frames_dir[int(x)])

    train_df.to_csv("../data/train.csv", index=False, columns=["exemplar", "frame_id", "label"])
    test_df.to_csv("../data/test.csv", index=False, columns=["exemplar", "frame_id", "label"])
    valid_df.to_csv("../data/validation.csv", index=False, columns=["exemplar", "frame_id", "label"])

    print("Train set: {}".format(len(train_df)))
    print("Test set: {}".format(len(test_df)))
    print("Validation set: {}".format(len(valid_df)))


def save_frames_to_use():
    frames_all = { f.name:f for f in fn.frames() }

    print("Total frames:", len(frames_all))

    parent_frames = set()

    print(frames_all['Traversing'].frameRelations.keys())

    # for f_name, frame in frames_all.items():




if __name__ == "__main__":
    save_frames_to_use()
