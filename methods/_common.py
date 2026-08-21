import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_clf(emb_dim, num_classes, hidden=1024, device="cpu"):
    return nn.Sequential(
        nn.Linear(emb_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    ).to(device)


def train_loop(clf, emb, lab, device, lr=1e-3, epochs=50, batch=64, extra_loss_fn=None):
    clf.train()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    emb, lab = emb.to(device), lab.to(device)
    n = len(lab)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s+batch]
            logits = clf(emb[idx])
            loss = F.cross_entropy(logits, lab[idx])
            if extra_loss_fn:
                loss = loss + extra_loss_fn()
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate(clf, emb, lab, device):
    clf.eval()
    emb, lab = emb.to(device), lab.to(device)
    return (clf(emb).argmax(1) == lab).float().mean().item()


def compute_metrics(R):
    T = len(R)
    avg_acc = np.mean(R[-1])
    if T <= 1:
        return avg_acc, 0.0, 0.0
    fgt = np.mean([max(R[i][j] for i in range(j, T-1)) - R[-1][j] for j in range(T-1)])
    bwt = np.mean([R[-1][j] - R[j][j] for j in range(T-1)])
    return avg_acc, fgt, bwt


def print_results(name, R):
    avg_acc, fgt, bwt = compute_metrics(R)
    print(f"\n{name}")
    print(f"Avg Acc: {avg_acc:.4f}")
    print(f"Forgetting: {fgt:.4f}")
    print(f"BWT: {bwt:.4f}")


class ReplayBuffer:
    """Fixed-size replay buffer with reservoir sampling."""
    def __init__(self, buffer_size=500):
        self.buffer_size = buffer_size
        self.emb = None
        self.lab = None
        self.logits = None  # for DER++
        self.n_seen = 0

    def update(self, emb, lab, logits=None):
        n = len(lab)
        if self.emb is None:
            if n <= self.buffer_size:
                self.emb = emb.clone().cpu()
                self.lab = lab.clone().cpu()
                self.logits = logits.clone().cpu() if logits is not None else None
            else:
                idx = torch.randperm(n)[:self.buffer_size]
                self.emb = emb[idx].clone().cpu()
                self.lab = lab[idx].clone().cpu()
                self.logits = logits[idx].clone().cpu() if logits is not None else None
            self.n_seen = n
        else:
            for i in range(n):
                self.n_seen += 1
                if len(self.lab) < self.buffer_size:
                    self.emb = torch.cat([self.emb, emb[i:i+1].cpu()])
                    self.lab = torch.cat([self.lab, lab[i:i+1].cpu()])
                    if logits is not None and self.logits is not None:
                        self.logits = torch.cat([self.logits, logits[i:i+1].cpu()])
                else:
                    j = torch.randint(0, self.n_seen, (1,)).item()
                    if j < self.buffer_size:
                        self.emb[j] = emb[i].cpu()
                        self.lab[j] = lab[i].cpu()
                        if logits is not None and self.logits is not None:
                            self.logits[j] = logits[i].cpu()

    def sample(self, batch_size, device="cpu"):
        idx = torch.randperm(len(self.lab))[:batch_size]
        result = (self.emb[idx].to(device), self.lab[idx].to(device))
        if self.logits is not None:
            return result + (self.logits[idx].to(device),)
        return result

    def __len__(self):
        return 0 if self.emb is None else len(self.lab)
