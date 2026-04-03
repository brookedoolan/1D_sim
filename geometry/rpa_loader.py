import numpy as np

def load_rpa_contour(filepath, n_points=None):
    """
    Load RPA contour text file.
    Returns axial position x (m), radius r (m).

    n_points: if provided, interpolate onto a uniform grid of this many points.
              e.g. n_points=500 gives dx ~ 0.6 mm for a 291 mm engine.
              Default (None) uses the raw RPA points (~177 points, dx ~ 1.65 mm).
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

    # Remove duplicate x points (e.g. RPA repeats the chamber/converging junction)
    #mask = np.concatenate(([True], np.diff(x) > 1e-9))
    #x = x[mask]
    #r = r[mask]

    if n_points is not None:
        x_fine = np.linspace(x[0], x[-1], n_points)
        r_fine = np.interp(x_fine, x, r)
        return x_fine, r_fine

    return x, r
