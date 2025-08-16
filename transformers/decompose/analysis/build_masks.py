import numpy as np
import pandas as pd
from preprocess_input import *


def char2token_idx(start_char, end_char, tokens):
    start_tokens = tokens.char_to_token(start_char)
    end_tokens = tokens.char_to_token(end_char)
    if start_tokens is None or end_tokens is None:
        print("Warning: char2token_idx is None, returning empty list")
        return []
    return np.arange(start_tokens, end_tokens + 1).tolist()


def split_arg_modifier_masks(t, tokenizer):
    arg = t["sentence"].split("<a>")[1]
    tokens = tokenizer(arg, add_special_tokens=False)
    stripped = t["stripped_arg"]

    arg_words = arg.split(" ")
    stripped_words = stripped.split(" ")
    in_stripped = []
    not_in_stripped = []
    modifier_token_idx = []

    for i, aw in enumerate(arg_words):
        found = False
        for sw in stripped_words:
            # correct for punctuation that might've been removed
            if sw.lower() in aw.lower() and len(sw)/len(aw) >= 0.8:
                in_stripped.append(aw)
                # remove to prevent using twice
                stripped_words.remove(sw)
                found = True
                break
        if not found and len(aw) > 0:
            not_in_stripped.append(aw)

            start_idx = len(" ".join(arg_words[:i]))
            if arg[start_idx] == " ":
                start_idx += 1

            # print("_" + arg[start_idx: start_idx + len(aw)] + "_")
            # print(arg[ start_idx + len(aw) - 1])
            token_idxs = char2token_idx(
                start_idx, start_idx + len(aw) - 1, tokens)
            # print(token_idxs)
            modifier_token_idx += token_idxs

    ids = np.array(tokens["input_ids"])
    modifier_mask = np.zeros_like(ids)
    modifier_token_idx = np.array(modifier_token_idx).astype(int)
    modifier_mask[modifier_token_idx] = 1
    # True if only in arg and not in arg_stripped, False otherwise
    modifier_mask = modifier_mask.astype(bool)

    # assert " ".join(in_stripped).strip() == tokenizer.decode(
    #     ids[~modifier_mask]).strip(), (
    #     " ".join(in_stripped).strip(),
    #     tokenizer.decode(ids[~modifier_mask]).strip()
    # )
    if " ".join(in_stripped).strip() != tokenizer.decode(
            ids[~modifier_mask]).strip():
        print("Warning! not equal:\t" + " ".join(in_stripped).strip(),
              tokenizer.decode(ids[~modifier_mask]).strip())
    # assert " ".join(not_in_stripped).strip(
    # ) == tokenizer.decode(ids[modifier_mask]).strip()

    if " ".join(not_in_stripped).strip() != tokenizer.decode(ids[modifier_mask]).strip():
        print("Warning! not equal:\t" + " ".join(not_in_stripped).strip(),
              tokenizer.decode(ids[modifier_mask]).strip())

    return modifier_mask


def get_arg_pred_masks(t, tokens, tokenizer):
    sentence = t["sentence"]
    ids = np.array(tokens["input_ids"])

    # argument
    # add 3 b/c <a> has length 3
    arg = sentence.split("<a>")[1]
    arg_start = sentence.index("<a>") + 3
    if sentence[arg_start] == " ":
        arg_start += 1
    arg_end = sentence.index("<a>", arg_start)
    arg_token_idxs = char2token_idx(arg_start, arg_end - 1, tokens)
    arg_token_idxs = np.array(arg_token_idxs)

    arg_mask = np.zeros_like(ids)
    arg_mask[arg_token_idxs] = 1
    # True iff in arg
    arg_mask = arg_mask.astype(bool)

    # assert arg.strip() == tokenizer.decode(ids[arg_mask]).strip()
    if arg.strip() != tokenizer.decode(ids[arg_mask]).strip():
        print("Warning! not equal:\t" + arg.strip(),
              tokenizer.decode(ids[arg_mask]).strip())

    mod_token_idxs = arg_token_idxs[np.where(t["modifier_mask"])]
    mod_mask = np.zeros_like(ids)
    mod_mask[mod_token_idxs] = 1
    mod_mask = mod_mask.astype(bool)

    arg_nomod_mask = arg_mask & ~mod_mask

    assert np.array_equal((mod_mask).astype(
        int) + (arg_nomod_mask).astype(int), arg_mask.astype(int))
    # raise

    if t["modifier_mask"].sum() != 0:
        print()
        print("arg: ", t["arg"])
        print("stripped arg: ", t["stripped_arg"])
        print("arg reconstructed (no mod): ",
              tokenizer.decode(ids[arg_nomod_mask]).strip())
        print("mod reconstructed: ", tokenizer.decode(ids[mod_mask]).strip())

    # predicate
    pred = sentence.split("<p>")[1]
    pred_start = sentence.index("<p>") + 3
    if sentence[pred_start] == " ":
        pred_start += 1
    pred_end = sentence.index("<p>", pred_start)
    pred_token_idxs = char2token_idx(pred_start, pred_end - 1, tokens)

    pred_mask = np.zeros_like(ids)
    pred_mask[pred_token_idxs] = 1
    # True iff in pred
    pred_mask = pred_mask.astype(bool)

    # assert pred.strip() == tokenizer.decode(ids[pred_mask]).strip()
    if pred.strip() != tokenizer.decode(ids[pred_mask]).strip():
        print("Warning! not equal:\t" + pred.strip(),
              tokenizer.decode(ids[pred_mask]).strip())

    return arg_nomod_mask.astype(int), mod_mask.astype(int), pred_mask.astype(int)


def get_other_masks(tokens, tokenizer, other_idxs):
    ids = np.array(tokens["input_ids"])
    other_mask = np.zeros_like(ids)
    for (start, end) in other_idxs:
        if end is None:
            continue
        other_token_idxs = char2token_idx(start, end - 1, tokens)
        other_mask[other_token_idxs] = 1
        print(tokenizer.decode(other_mask * ids))
    return other_mask


prn_inflect = {
    "i": "me",
    "me": "I",
    "he": "him",
    "him": "he",
    "she": "her",
    "her": "she",
    "we": "us",
    "us": "we",
    "they": "them",
    "them": "they"
}


def build_masks(dataframe, tokenizer):
    dataset = []
    lengths = []

    for row in dataframe.iloc:
        try:
            sentence, arg = format_input(
                row["sentence"],
                eval(row["arg_idx"]),
                eval(row["verb_idx"]))
            lengths.append(len(tokenizer(sentence).input_ids))
            structure = row["structure"]

            stripped_arg = fix_punct(row["Arg.Stripped"])
            if "passive" in structure:
                stripped_arg = [prn_inflect[x.lower()] if x.lower() in prn_inflect
                                else x for x in stripped_arg.split(" ")]
                stripped_arg = " ".join(stripped_arg)

            dataset.append({
                "sentence": sentence, "index": row['index'], "arg": arg,
                "stripped_arg": stripped_arg, "gram": row["Gram.Func"],
                "structure": row["structure"]})
        except ValueError as e:
            print(e)

    for t in dataset:
        tokens = tokenizer(t["sentence"], add_special_tokens=True)
        t['modifier_mask'] = split_arg_modifier_masks(
            t, tokenizer).astype(int)
        t['arg_nomod_mask'], t['arg_mod_mask'], t['pred_mask'] = get_arg_pred_masks(
            t, tokens, tokenizer)

    df = pd.DataFrame(dataset)
    df = df.melt(id_vars=["index", "sentence", "structure", "gram"], value_name="mask", value_vars=[
        "arg_nomod_mask", "arg_mod_mask", "pred_mask"], var_name="contribution")

    return {
        "dataset": dataset,
        "long_dataframe": df,
        "token_lengths": lengths
    }


def find_remove_marks(sentence, start_mark, end_mark):
    idxs = []
    while start_mark in sentence:
        sid = sentence.index(start_mark)
        eid = sentence.index(end_mark)
        # subtract 1 b/c sid gets deleted
        idxs.append((sid, eid - 1))
        sentence = sentence[:sid] + sentence[sid + 1: eid] + sentence[eid + 1:]
    return sentence, idxs


def build_masks_with_other(dataframe, tokenizer):
    dataset = []
    lengths = []

    dataframe["other_args_idx"] = dataframe.apply(
        lambda row: [
            eval(x) for x in dataframe.loc[dataframe["sentence"] == row["sentence"]]["arg_idx"].tolist()
            if x != row["arg_idx"]],
        axis=1)

    for row in dataframe.iloc:
        try:
            sentence, arg, other_args = format_input_with_other(
                row["sentence"],
                eval(row["arg_idx"]),
                eval(row["verb_idx"]),
                row["other_args_idx"])

            sentence, other_idxs = find_remove_marks(
                sentence, start_mark="Ø", end_mark="ø")
            lengths.append(len(tokenizer(sentence).input_ids))
            structure = row["structure"]

            stripped_arg = fix_punct(row["Arg.Stripped"])
            if "passive" in structure:
                stripped_arg = [prn_inflect[x.lower()] if x.lower() in prn_inflect
                                else x for x in stripped_arg.split(" ")]
                stripped_arg = " ".join(stripped_arg)

            dataset.append({
                "sentence": sentence, "index": row['index'], "arg": arg,
                "stripped_arg": stripped_arg, "gram": row["Gram.Func"],
                "structure": row["structure"], "other_args": other_args,
                "other_idxs": other_idxs})
        except ValueError as e:
            print(e)

    for t in dataset:
        tokens = tokenizer(t["sentence"], add_special_tokens=True)
        t['modifier_mask'] = split_arg_modifier_masks(
            t, tokenizer).astype(int)
        t['arg_nomod_mask'], t['arg_mod_mask'], t['pred_mask'] = get_arg_pred_masks(
            t, tokens, tokenizer)
        t['other_args_mask'] = get_other_masks(
            tokens, tokenizer, t["other_idxs"])

    df = pd.DataFrame(dataset)
    df = df.melt(id_vars=["index", "sentence", "structure", "gram"], value_name="mask", value_vars=[
        "arg_nomod_mask", "arg_mod_mask", "pred_mask", "other_args_mask"], var_name="contribution")

    return {
        "dataset": dataset,
        "long_dataframe": df,
        "token_lengths": lengths
    }
