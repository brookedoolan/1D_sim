import numpy as np

def load_rpa_contour(filepath):
    """
    Load RPA contour text file.
    Returns axial position x (m), radius r (m)
    """

    x_vals = []
    r_vals = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and header lines
            if line.startswith("#") or line == "":
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    x_mm = float(parts[0])
                    r_mm = float(parts[1])

                    x_vals.append(x_mm*1e-3)
                    r_vals.append(r_mm*1e-3)

                except ValueError:
                    continue
            
    x = np.array(x_vals)
    r = np.array(r_vals)

    return x, r