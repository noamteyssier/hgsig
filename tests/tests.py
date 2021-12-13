"""
Testing suite for `hgsig`
"""

import numpy as np
from hgsig import HGSig

def main():
    np.random.seed(42)
    num = 10000
    num_x = 3000
    num_c = 5
    num_g = 10

    reference = ["g0", "g2"]
    clusters = np.array([f"c{i}" for i in np.random.choice(num_c, size=num + num_x)])
    groups = np.array([f"g{i}" for i in np.random.choice(num_g, size=num)])
    groups = np.concatenate([
        groups,
        [np.random.choice(reference) for _ in range(num_x)]
        ])

    hgs = HGSig(
        clusters,
        groups,
        reference,
        method="fishers")
    hgs.run()

if __name__ == "__main__":
    main()
