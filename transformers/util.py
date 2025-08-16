import re
import numpy as np

properties_list = ['awareness', 'change_of_location', 'change_of_state',
       'change_of_possession', 'existed_after', 'existed_before',
       'existed_during', 'instigation', 'sentient', 'volition']


def make_labels(x):
    x = np.array(x)
    labels = np.ones(len(properties_list))
    labels[np.where(x <= 2)] = 0
    labels[np.where(x >= 4)] = 2
    return labels

def fix_punct(s):
    s = s.replace("\\", "").replace("''", '"').replace("``", '"').replace(" s ", "s ")
    # insert a space after comma or period in non-number
    s = re.sub(r'(\.|,)([^0-9])', r'\1 \2', s)
    # insert a space between groups of characters and letters
    s = re.sub(r'(\d+)([A-z]+)', r'\1 \2', s)
    s = re.sub(r'([A-z]+)(\d+)', r'\1 \2', s)
    # fix " "
    s = re.sub(r'\s*"\s*(.*?)\s*"\s*', r' "\1" ', s).strip()
    # unclosed final "
    s = re.sub(r'\s*"\s*([^"]*?)', r' "\1 ', s).strip()

    # remove traces
    s = " ".join(list(filter(lambda x: "*" not in x, s.split(" "))))
    s = s.replace("-LRB-", "(").replace("-RRB-", ")").replace("-LCB-", "{"
        ).replace("-RCB-", "}").replace("`", "'").replace(" n't", "n't"
        ).replace('."', '".').replace(".'", "'.").replace(',"', '",')

    # fix punctuation
    for p in ";:,.'?!%-){}+/":
        s = s.replace(" " + p, p)
    for p in "#$-({}/ ":
        s = s.replace(p + " ", p)
    return s
