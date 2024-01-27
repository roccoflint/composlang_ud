import typing
from sys import stderr
from pathlib import Path


def log(*args, **kwargs):
    """
    placeholder logging method to be changed later
    """
    print("** info:", *args, **kwargs, file=stderr)


def pathify(fpth: typing.Union[Path, str, typing.Any]) -> Path:
    """
    returns a resolved `Path` object after expanding user and shorthands/symlinks
    """
    return Path(fpth).expanduser().resolve()


def iterable_from_directory_or_filelist(directory_or_filelist) -> typing.Iterable[Path]:
    """
    constructs an iterable over Path objects using a given directory, file, or
    list of files as either str or Path objects
    """
    # either it must be a list of files, or a directory containing files, or a file
    # first, assume it is a directory or individual file
    try:
        directory_or_filelist = pathify(directory_or_filelist)
        if directory_or_filelist.is_dir():
            files = [*directory_or_filelist.iterdir()]
        else:
            files = [directory_or_filelist]
    # next, assume we are given a list of filepaths as strs or or Path-like objects
    except TypeError as e:
        files = list(map(pathify, directory_or_filelist))

    return files


def minmax(a):
    return (a - a.min()) / (a.max() - a.min())


def ECDF_transform(a):
    from statsmodels.distributions.empirical_distribution import ECDF

    return ECDF(a)(a)


def get_llm_results(
    model: str,
    study: str,
    basedir: Path = Path("./llm-results/"),
    adj_freq: typing.Dict[str, int] = None,
    noun_freq: typing.Dict[str, int] = None,
    pair_freq: typing.Dict[typing.Tuple[str, str], int] = None,
):
    """ """
    import pandas as pd
    import numpy as np
    from functools import reduce

    # log(model, study, paradigm)
    assert all(
        x is not None for x in (adj_freq, noun_freq, pair_freq)
    ), "missing frequency data"

    [resultsdir] = (basedir / study).glob("benchmark-cfg=*")

    results = []
    for paradigm in ["logprobs", "likert"]:
        results_csvs = list(resultsdir.glob(f"eval={paradigm}*/model={model}/*.csv"))
        if paradigm == "logprobs":
            results_paradigm = [
                pd.read_csv(r_path, index_col=0) for r_path in results_csvs
            ]
        elif paradigm == "likert":
            results_paradigm = []
            for r_path in results_csvs:
                r = pd.read_csv(r_path, index_col=0)
                likert_kind = r_path.parts[3].split("=")[1]
                r.rename(
                    columns={"text_likert_noun_adjective": likert_kind}, inplace=True
                )
                results_paradigm.append(r)
        else:
            raise ValueError(f"Unrecognized paradigm: {paradigm}")

        results += results_paradigm

    if len(results) > 1:
        results = reduce(pd.merge, results)
    elif len(results) == 1:
        results = results[0]
    else:
        raise ValueError(f"Found {len(results)} results")

    # any column header that is the result of model outputs
    model_columns = [
        "likert_constrained_original",
        "likert_constrained_optimized",
        "logp_A",
        "logp_N",
        "logp_N_A",
        "logp_AN",
    ]

    metadata_columns = results.columns.difference(model_columns)

    # construct multiindex to dilineate between metadata and model columns
    # SIDE-NOTE: WHY is this necessary? there is no overlap in the kinds of data across the two
    mii = pd.MultiIndex.from_tuples(
        tuples=[("metadata", c) for c in results.columns if c in metadata_columns]
        + [("model", c) for c in results.columns if c in model_columns],
        # names=levels,
    )
    results.columns = mii
    results["metadata", "model"] = model

    # these are frequencies provided independently, calculated using some corpus (e.g. COCA)
    results["metadata", "adj_freq"] = (
        results["metadata", "adjective"].apply(str.lower).map(adj_freq)
    )
    results["metadata", "noun_freq"] = (
        results["metadata", "noun"].apply(str.lower).map(noun_freq)
    )
    results["metadata", "pair_freq"] = (
        (results[("metadata", "adjective")] + " " + results[("metadata", "noun")])
        .apply(str.lower)
        .map(pair_freq)
    )
    results.fillna(0, inplace=True)

    if paradigm == "logprobs":
        for col in ["logp_N_A", "logp_AN"]:
            col = "logp_N_A"
            ecdf = ECDF_transform(results["model", col])
            results["model", f"ecdf_{col}"] = ecdf

            if col == "logp_N_A":
                conditionals = results["model", col]
                # conditionals are in log space, and adj_freq is absolute counts.
                # rescale to sum of counts of all adjs (not just those in the dataset)
                hybrid_p = conditionals + np.log(
                    results["metadata", "adj_freq"] / sum(adj_freq.values())
                )
                results["model", f"corpus_{col}"] = hybrid_p
    if paradigm == "likert":
        for col in ["likert_constrained_original", "likert_constrained_optimized"]:
            results["model", f"{col}"] -= 4

    return results.sort_index(axis=1)
