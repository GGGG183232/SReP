import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict, OrderedDict
from rdchiral.main import rdchiralRunText


def preprocess(X, fp_dim):
    mol = Chem.MolFromSmiles(X)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=int(fp_dim), useChirality=True)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits())
    arr[onbits] = 1
    return arr


def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, ts = zip(*l)
        ret.append((reactant, sum(ss), list(ts)[0]))
    reactants, scores, templates = zip(*sorted(ret, key=lambda item: item[1], reverse=True))
    return list(reactants), list(scores), list(templates)


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=512, dropout_rate=0.3):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(dim, n_rules)

    def forward(self, x):
        x = self.fc3(self.dropout1(self.bn1(self.fc1(x))))
        return x


def load_parallel_model(state_path, template_rule_path, fp_dim=2048):
    template_rules = {}
    with open(template_rule_path, 'r') as f:
        for i, l in tqdm(enumerate(f), desc='template rules'):
            rule = l.strip()
            template_rules[rule] = i
    idx2rule = {}
    for rule, idx in template_rules.items():
        idx2rule[idx] = rule
    rollout = RolloutPolicyNet(len(template_rules), fp_dim=fp_dim)
    checkpoint = torch.load(state_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name] = v
    rollout.load_state_dict(new_state_dict)
    return rollout, idx2rule


class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        self.net, self.idx2rules = load_parallel_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        self.net.to(device)

    def run(self, x, topk=10):
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr, [-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        arr = arr.to(self.device)
        preds = self.net(arr)
        preds = F.softmax(preds, dim=1)
        preds = preds.cpu()
        probs, idx = torch.topk(preds, k=topk)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        templates = []
        for i, rule in enumerate(rule_k):
            try:
                out1 = rdchiralRunText(rule, x)
                if len(out1) == 0:
                    continue
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    templates.append(rule)
            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                pass
        if len(reactants) == 0:
            return None
        reactants_d = defaultdict(list)
        for r, s, t in zip(reactants, scores, templates):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, t))
            else:
                reactants_d[r].append((s, t))
        reactants, scores, templates = merge(reactants_d)
        total = sum(scores)
        scores = [s / total for s in scores]
        return {'reactants': reactants,
                'scores': scores,
                'template': templates}

