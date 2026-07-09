import numpy as np


def read_track_state_data(filename):
    """
    Each line:

    x1, y1, z1, x2, y2, z2, x3, y3, z3, tof,
    p, theta, phi

    Returns
    -------
    features : np.ndarray
        shape [N, 10]
        (x1, y1, z1, x2, y2, z2, x3, y3, z3, tof)

    targets : np.ndarray
        shape [N, 3]
        (p, theta, phi)
    """

    features = []
    targets = []

    with open(filename, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() != ""
        ]

    for i, line in enumerate(lines):
        vals = [float(x) for x in line.split(",")]

        if len(vals) != 13:
            raise ValueError(
                f"Line {i+1}: expected 13 values "
                f"(10 input + particleType + 3 output), "
                f"got {len(vals)}"
            )

        (
            x1, y1, z1, x2, y2, z2, x3, y3, z3, tof,
            p, theta, phi
        ) = vals

        features.append([x1, y1, z1, x2, y2, z2, x3, y3, z3, tof,])

        targets.append([p, theta, phi])

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    return features, targets


# Example usage
if __name__ == "__main__":

    features, targets = read_track_state_data(
        "track_state.csv"
    )

    print(f"Read {len(features)} samples")

    print("\nInput shape :", features.shape)
    print("Target shape:", targets.shape)

    print("\nFirst sample:")
    print("input  =", features[0])
    print("target =", targets[0])