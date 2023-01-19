import typing
from pathlib import Path
import json

import pandas as pd


def process_subject(path: typing.Union[str, Path]):
    """
    given a path to a jspsych experiment saved data, processes the subject to extract a
    dataframe with only relevant information retaiend and properly organized
    """
    with Path(path).open("r") as f:
        data = json.load(f)

    metadata = {}
    lowercased_dict_keys = lambda dikt: {k.lower(): v for k, v in dikt.items()}

    metadata["subject_id"] = data[0]["subject_id"]
    metadata["study_id"] = data[0]["study_id"]
    for i in (1, 2):
        metadata.update(lowercased_dict_keys(data[i]["response"]))
    try:
        metadata["age"] = int(metadata["age"])
    except ValueError:
        metadata["age"] = float("nan")

    frames = []
    skipped = 0
    for i in range(len(data)):
        if data[i]["trial_type"] != "survey-likert":
            skipped += 1
            continue
        frame = {}
        frame.update(metadata)

        for k in ("rt", "time_elapsed", "trial_index"):
            frame[k] = data[i][k]
        [frame["item"]] = [*data[i]["response"].keys()]
        [frame["response"]] = [*data[i]["response"].values()]
        frame["response"] += -3
        frame["trial_index"] += -skipped
        if frame["item"].startswith("gold_"):
            frame["item"] = frame["item"][len("gold_") :]
            frame["trial_type"] = "gold"
        elif frame["item"].startswith("anti_"):
            frame["item"] = frame["item"][len("anti_") :]
            frame["trial_type"] = "anti"
        else:
            frame["trial_type"] = "crit"

        frames += [frame]

    return pd.DataFrame(frames)


def find_subject(
    subject_id, basedir=Path("real-deal/").expanduser().resolve(), search_all=True
):
    processed = []
    for path in basedir.glob("*.json"):
        with Path(path).open("r") as f:
            data = json.load(f)
            if subject_id == data[0]["subject_id"]:
                processed += [process_subject(path)]
                if search_all:
                    continue
                else:
                    break
    if processed:
        return pd.concat(processed)
    raise FileNotFoundError(f"no response for subject `{subject_id}` at `{basedir}`")


def compile_data(basedir="real-deal/"):
    basedir = Path(basedir).expanduser().resolve()
    glob = [*basedir.glob("*.json")]
    df = pd.concat(process_subject(sub) for sub in glob).reset_index(drop=True)

    subject_to_ix = {}  # dict((item, i) for i, item in enumerate(set(df.subject_id)))
    # item_to_ix = dict((item, i) for i, item in enumerate(set(df.item)))
    item_to_ix = {}
    for i, row in df[["item", "trial_type", "subject_id"]].iterrows():
        item = row["item"]
        tt = row["trial_type"]
        sub = row["subject_id"]
        if item not in item_to_ix:
            item_to_ix[item] = len(item_to_ix)
        if sub not in subject_to_ix:
            subject_to_ix[sub] = len(subject_to_ix)

    ix_to_item = {v: k for k, v in item_to_ix.items()}
    ix_to_subject = {v: k for k, v in subject_to_ix.items()}

    assert len(item_to_ix) == len(ix_to_item)
    assert len(subject_to_ix) == len(ix_to_subject)

    df["item_index"] = df["item"].apply(lambda item: item_to_ix[item])

    return df, subject_to_ix, ix_to_subject, item_to_ix, ix_to_item
