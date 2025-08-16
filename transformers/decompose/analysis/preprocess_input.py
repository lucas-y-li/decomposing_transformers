import re
import numpy as np


def make_labels_onehot(x, num_properties):
    x = np.array(x)
    labels = np.zeros((3, num_properties))
    # 1 or 2 (negative)
    labels[0, :] = (x <= 2)
    labels[1, :] = np.abs(x - 3) < 1
    labels[2, :] = (x >= 4)
    return labels.flatten()


def make_labels(x, num_properties):
    x = np.array(x)
    labels = np.ones(num_properties)
    labels[np.where(x <= 2)] = 0
    labels[np.where(x >= 4)] = 2
    return labels


def fix_punct(s):
    s = s.replace("\\", "").replace("''", '"').replace(
        "``", '"').replace(" s ", "s ")
    # replace '  with "  when used as a quotation
    s = re.sub(r"(^|>|\s)' ", r"\1'' ", s)
    s = re.sub(r" '($|<|\s)", r" ''\1", s)
    s = re.sub(r"\s*''\s*(.*?)\s''\s*", r" '\1' ", s).strip()

    # insert a space after comma or period in non-number
    s = re.sub(r'(\.|,)([^0-9])', r'\1 \2', s)
    # insert a space between groups of characters and letters
    s = re.sub(r'(\d+)([A-z]+)', r'\1 \2', s)
    s = re.sub(r'([A-z]+)(\d+)', r'\1 \2', s)
    # fix " " and ' '
    s = re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', s).strip()
    # unclosed final "
    s = re.sub(r'\s+"\s+([^"]+?)$', r' "\1', s).strip()

    # remove traces
    s = " ".join(list(filter(lambda x: "*" not in x, s.split(" "))))
    s = s.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LCB-", "{"
                                                              ).replace("-RCB-", "}").replace("`", "'").replace(" n't", "n't"
                                                                                                                ).replace('."', '".').replace(".'", "'.").replace(',"', '",')

    # fix punctuation
    for p in ";:.,?!%-){}+/":
        s = s.replace(" " + p, p)
    for p in "#$-({}/ ":
        s = s.replace(p + " ", p)
    # fix 's and 't
    s = re.sub(r" '(s|t)(\s|<|$)", r"'\1\2", s)
    s = s.replace(". com", ".com")
    return s


def format_input(sentence, arg_idx, pred_idx):
    words = sentence.split(" ")

    arg = words[arg_idx[0]: arg_idx[1]]
    arg = fix_punct(" ".join(arg))

    if arg_idx[0] == 0:
        words[arg_idx[0]] = "<a>" + words[arg_idx[0]]
    else:
        words[arg_idx[0]] = "<a> " + words[arg_idx[0]]
    words[arg_idx[1] - 1] = words[arg_idx[1] - 1] + "<a>"
    if pred_idx[0] == 0:
        words[pred_idx[0]] = "<p>" + words[pred_idx[0]]
    else:
        words[pred_idx[0]] = "<p> " + words[pred_idx[0]]
    words[pred_idx[1] - 1] = words[pred_idx[1] - 1] + "<p>"

    sentence = " ".join(words)
    sentence = fix_punct(fix_punct(sentence)).replace(
        " <p>", "<p>").replace(" <a>", "<a>")

    return sentence, arg


def format_input_with_other(sentence, arg_idx, pred_idx, other_args_idx):
    assert "ø" not in sentence
    assert "Ø" not in sentence

    words = sentence.split(" ")

    arg = words[arg_idx[0]: arg_idx[1]]
    arg = fix_punct(" ".join(arg))

    if arg_idx[0] == 0:
        words[arg_idx[0]] = "<a>" + words[arg_idx[0]]
    else:
        words[arg_idx[0]] = "<a> " + words[arg_idx[0]]
    words[arg_idx[1] - 1] = words[arg_idx[1] - 1] + "<a>"

    other_args = []
    for other_idx in other_args_idx:
        other_arg = words[other_idx[0]: other_idx[1]]
        other_arg = fix_punct(" ".join(other_arg))
        other_args.append(arg)

        words[other_idx[0]] = "Ø" + words[other_idx[0]]
        words[other_idx[1] - 1] = words[other_idx[1] - 1] + "ø"

    if pred_idx[0] == 0:
        words[pred_idx[0]] = "<p>" + words[pred_idx[0]]
    else:
        words[pred_idx[0]] = "<p> " + words[pred_idx[0]]
    words[pred_idx[1] - 1] = words[pred_idx[1] - 1] + "<p>"

    sentence = " ".join(words)
    sentence = fix_punct(fix_punct(sentence)).replace(
        " <p>", "<p>").replace(" <a>", "<a>")

    return sentence, arg, other_args


def split_pos_neg_contributions(logits):
    """
    shape: (num_contributions + 1 (bias), samples, num_classes, num_labels)
    """
    # put "neutral" category into "negative"

    logits[..., 0, :] += logits[..., 1, :]
    positive_mask = (logits > 0).astype(int)

    # put negative "positive" contributions in the negative category
    # put negative "negative" contributions in the positive category

    positive_logits = logits[..., 2, :] * positive_mask[..., 2, :]
    positive_logits -= logits[..., 0, :] * (1 - positive_mask[..., 0, :])

    negative_logits = logits[..., 0, :] * positive_mask[..., 0, :]
    negative_logits -= logits[..., 2, :] * (1 - positive_mask[..., 2, :])

    assert (np.all(positive_logits >= 0))
    assert (np.all(negative_logits >= 0))

    binary_logits = np.stack(
        [negative_logits, positive_logits], axis=-2
    )
    print(binary_logits.shape)
    return binary_logits
