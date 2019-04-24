import pandas as pd

from nltk.corpus import framenet as fn

from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes

from collections import defaultdict

import csv


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

    train_df.to_csv("../data/exemplars_train.csv", index=False, columns=["exemplar", "frame_id", "label"])
    test_df.to_csv("../data/exemplars_test.csv", index=False, columns=["exemplar", "frame_id", "label"])
    valid_df.to_csv("../data/exemplars_validation.csv", index=False, columns=["exemplar", "frame_id", "label"])

    print("Train set: {}".format(len(train_df)))
    print("Test set: {}".format(len(test_df)))
    print("Validation set: {}".format(len(valid_df)))


def get_docs():

    docs_to_use, docs_to_ignore = [], []

    docs = fn.docs()

    for doc in docs:
        print("working on", doc.name)
        has_targets = False
        for sentence in doc.sentence:
            if sentence.targets:
                has_targets = True
                break

        if has_targets:
            docs_to_use.append((doc.ID, doc.name))
        else:
            docs_to_ignore.append((doc.ID, doc.name))

    return docs_to_use, docs_to_ignore


def to_csv(file_path, csv_data, csv_header):
    with open(file_path, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(csv_header)
        writer.writerows(csv_data)


def framenet_docs():
    docs_to_use, docs_to_ignore = get_docs()
    csv_header = ["doc_id", "doc_name"]
    to_csv("../misc/docs_to_use.csv", docs_to_use, csv_header)
    to_csv("../misc/docs_to_ignore.csv", docs_to_ignore, csv_header)


def gen_input_csv(input_file, output_path, frames_header, frame_label):

    docs = pd.read_csv(input_file)

    total_frames = len(frames_header)

    content = []
    for idx, row in docs.iterrows():
        doc = fn.doc(row.doc_id)
        for sentence in doc.sentence:
            example = [sentence.text]
            labels = [0] * total_frames
            for seq, predicate, frame in sentence.targets:
                labels[frame_label[frame]] = 1
            content.append(example + labels)

    to_csv(output_path, content, frames_header)


def fn_docs_train_test_val():
    train_docs, val_docs, test_docs = ["../misc/docs_for_{}.csv".format(x) \
                                       for x in ["train", "dev", "test"]]

    train_path, val_path, test_path = ["../data/ft_{}.csv".format(x) \
                                       for x in ["train", "val", "test"]]

    frames_df = pd.read_csv("../misc/frames_ft.csv")

    frames_header = [row.frame for _, row in frames_df.iterrows()]
    frame_label = {row.frame: row.label for _, row in frames_df.iterrows()}

    print("Generating train files...")
    gen_input_csv(train_docs, train_path, frames_header, frame_label)
    print("Generating test files...")
    gen_input_csv(test_docs, test_path, frames_header, frame_label)
    print("Generating val files...")
    gen_input_csv(val_docs, val_path, frames_header, frame_label)




def get_frames_used(input="../misc/docs_to_use.csv", output="../misc/frames_ft.csv"):
    frames = defaultdict(lambda: 0)

    docs = pd.read_csv(input)

    for idx, row in docs.iterrows():
        doc = fn.doc(row.doc_id)
        for sentence in doc.sentence:
            for seq, predicate, frame in sentence.targets:
                frames[frame] += 1

    frames_list = [(label, frame, count) for label, (frame, count) in enumerate(frames.items())]
    frame_header = ["label", "frame", "count"]

    to_csv(output, frames_list, frame_header)




def preprocess_ontonotes():
    import logging
    logging.basicConfig(level=logging.INFO)
    ontonotes = Ontonotes()
    ontonotes_path = "../OntoNotes/conll-formatted-ontonotes-5.0/"
    ontonotes_dataset = ontonotes.dataset_iterator(ontonotes_path)
    for dataset in ontonotes_dataset:
        print(dataset.keys())


if __name__ == "__main__":
    fn_docs_train_test_val()
