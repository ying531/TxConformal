import numpy as np

def make_score(name="clip", M=100.0):
    if name == "residual_c":
        # y ∈ {0,1}
        def score(y, f, c):
            return c - f 
        return score
    elif name == "clip":
        # regression w/ threshold c (works for binary too if c=0)
        def score(y, f, c):
            return (M * (y > c).astype(float) + c * (y <= c).astype(float)) - f
        return score
    elif name == "residual":
        def score(y, f, c):
            return y - f
        return score
    else:
        raise ValueError("Unknown score")