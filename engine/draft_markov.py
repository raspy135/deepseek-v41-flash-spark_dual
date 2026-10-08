"""Greedy draft-only candidate scoring; never used for target verification."""
import torch

def shortlist_greedy(logits, embed, head, anchor, k, linear):
    """Return a proposal chain and previous-token embeddings for its confidence head.

    A candidate removed before applying its Markov bias cannot win this proposal.
    This is deliberately approximate and must be judged by acceptance and tok/s,
    not merely the much smaller weight read. Both ranks run the same static path.
    """
    ids = logits.topk(k, dim=-1).indices.sort(dim=-1).values
    values = logits.gather(1, ids)
    prev = anchor.reshape(1)
    proposals, embeddings = [], []
    for i in range(logits.shape[0]):
        e = embed[prev]
        embeddings.append(e)
        bias = linear(e, head[ids[i]]).float()[0]
        best = (values[i] + bias).argmax().view(1)
        # A scalar CUDA tensor index can force .item() during graph capture.
        prev = ids[i].gather(0, best)
        proposals.append(prev)
    return torch.cat(proposals), embeddings
