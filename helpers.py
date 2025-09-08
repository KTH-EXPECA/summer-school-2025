# Import required packages
import pickle, json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# non conditional training helper function
def train_slp_gmm_model(
    num_centers,
    df,
    num_samples,
    batch_size,
    num_epochs,
    learning_rate,
    standardize,
    val_fraction=0.2,   # fraction of samples for validation
):

    model = GaussianMM(centers=num_centers)

    # --- Prepare data ---
    delays = pd.to_numeric(df['packet_delay_ms'], errors='coerce').dropna().to_numpy()
    Y = np.asarray(delays, dtype=np.float64).reshape(-1, 1)

    if standardize:
        y_mean = float(Y.mean())
        y_std  = float(Y.std() + 1e-8)   # avoid divide-by-zero
    else:
        y_mean = 0.0
        y_std  = 1.0

    Yz = (Y - y_mean) / y_std

    # Ensure shapes are (N,1)
    Xz = np.zeros((len(Yz), 1))
    Yz = np.asarray(Yz).reshape(-1, 1)

    # pick a subset of training samples
    idx = np.random.choice(len(Yz), size=num_samples, replace=False)
    X_sub = Xz[idx]
    Y_sub = Yz[idx]

    # --- Train/validation split ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_sub, Y_sub, test_size=val_fraction, random_state=42
    )

    # --- Training rounds ---
    training_rounds = [{"learning_rate": learning_rate, "epochs": num_epochs}]

    for i, rp in enumerate(training_rounds, 1):
        print(f"Training session {i}/{len(training_rounds)} with {rp}")

        model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=rp["learning_rate"]),
            loss=model.loss,
        )

        history = model.training_model.fit(
            x=[X_train, Y_train],
            y=Y_train,
            batch_size=batch_size,
            epochs=rp["epochs"],
            verbose=1,
            shuffle=True,
            validation_data=([X_val, Y_val], Y_val),
        )

    return model, y_mean, y_std, history


def plot_ccdf_pdf_with_model(
    df,
    model,
    max_x: float = 400.0,
    num_points: int = 1200,
    batch_size: int = 2048,
    y_mean: float | None = None,
    y_std: float | None = None,
    pdf_bins: int = 400,
    use_kde: bool = False,
):
    """
    delays_ms : 1D array of delays (ms)
    model     : your GaussianMM instance (with .prob_pred_model)
    y_mean,y_std : if you trained on standardized targets, pass them here
    """

    # ----- Clean & sort data -----
    delays = pd.to_numeric(df['packet_delay_ms'], errors='coerce').dropna().to_numpy()
    delays = delays[(delays >= 0) & (delays <= max_x)]
    delays.sort()
    n = delays.size
    if n == 0:
        raise ValueError("No valid delay samples in range.")

    # ----- Empirical CCDF -----
    emp_ccdf = 1.0 - (np.arange(1, n + 1) / n)

    # ----- Grid for model curves -----
    grid_y = np.linspace(0.0, max_x, num_points)

    # If trained on z, feed z into the model
    if (y_mean is not None) and (y_std is not None):
        z = (grid_y - y_mean) / (y_std + 1e-12)
        jac = 1.0 / (y_std + 1e-12)  # for PDF only
    else:
        z = grid_y
        jac = 1.0

    # dtype & shapes for model inputs
    try:
        dtype_np = getattr(model, "dtype").as_numpy_dtype
    except Exception:
        dtype_np = np.float64

    X_dummy = np.zeros_like(z, dtype=dtype_np).reshape(-1, 1)
    Y_in    = np.asarray(z, dtype=dtype_np).reshape(-1, 1)

    # Predict model pdf/logpdf/ecdf
    pdf_m, logpdf_m, ecdf_m = model.prob_pred_model.predict(
        [X_dummy, Y_in], batch_size=batch_size, verbose=0
    )
    pdf_m   = np.squeeze(pdf_m)   * jac         # scale pdf if standardized
    ccdf_m  = 1.0 - np.squeeze(ecdf_m)         # ccdf unaffected by affine scale

    # ----- Empirical PDF (KDE or histogram density) -----
    # Try KDE for smooth curve; fallback to histogram
    emp_x_pdf = None
    emp_pdf   = None
    if use_kde:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(delays)
            emp_x_pdf = grid_y
            emp_pdf   = kde(emp_x_pdf)
        except Exception:
            pass

    if emp_pdf is None:
        # Histogram density estimate
        counts, edges = np.histogram(delays, bins=pdf_bins, range=(0, max_x), density=True)
        emp_x_pdf = 0.5 * (edges[:-1] + edges[1:])
        emp_pdf   = counts

    # ===================== PLOTS =====================

    # CCDF
    plt.figure(figsize=(5, 3))
    plt.step(delays, emp_ccdf, where='post', label='Empirical CCDF')
    plt.plot(grid_y, ccdf_m, label='Predicted CCDF')
    plt.yscale('log')
    plt.xlim(0, max_x)
    plt.ylim(1e-5, 1)
    plt.xlabel('Packet delay (ms)')
    plt.ylabel('CCDF  (P[Delay > x])')
    plt.title('Packet Delay CCDF: Data vs Model')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PDF
    plt.figure(figsize=(5, 3))
    # empirical (step if hist; line if KDE)
    if len(emp_x_pdf) == len(grid_y):
        plt.plot(emp_x_pdf, emp_pdf, label='Empirical PDF')
    else:
        plt.step(emp_x_pdf, emp_pdf, where='mid', label='Empirical PDF')
    # model
    plt.plot(grid_y, pdf_m, label='Predicted PDF')
    plt.xlim(0, max_x)
    plt.xlabel('Packet delay (ms)')
    plt.ylabel('PDF')
    plt.title('Packet Delay PDF: Data vs Model')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()



from pr3d.de import GaussianMM, GaussianMixtureEVM, GammaMixtureEVM, AppendixEVM
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# training function
def train_slp_evm_model(
    centers,
    df,
    num_samples,
    batch_size,
    bulk_num_epochs,
    bulk_learning_rate,
    evm_num_epochs,
    evm_learning_rate,
    standardize,
    val_fraction=0.2,   # fraction of samples for validation
):

    # --- Prepare data ---
    delays = pd.to_numeric(df['packet_delay_ms'], errors='coerce').dropna().to_numpy()
    Y = np.asarray(delays, dtype=np.float64).reshape(-1, 1)

    if standardize:
        y_mean = float(Y.mean())
        y_std  = float(Y.std() + 1e-8)   # avoid divide-by-zero
    else:
        y_mean = 0.0
        y_std  = 1.0

    Yz = (Y - y_mean) / y_std

    # Ensure shapes are (N,1)
    Xz = np.zeros((len(Yz), 1))
    Yz = np.asarray(Yz).reshape(-1, 1)

    # pick a subset of training samples
    idx = np.random.choice(len(Yz), size=num_samples, replace=False)
    X_sub = Xz[idx]
    Y_sub = Yz[idx]

    # --- Train/validation split ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_sub, Y_sub, test_size=val_fraction, random_state=42
    )


    # --- 1) Bulk Training rounds ---
    training_rounds = [{"learning_rate": bulk_learning_rate, "epochs": bulk_num_epochs}]

    bulk_model = GaussianMM(centers=centers)

    for i, rp in enumerate(training_rounds, 1):
        print(f"Training session {i}/{len(training_rounds)} with {rp}")

        bulk_model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=rp["learning_rate"]),
            loss=bulk_model.loss,
        )

        history = bulk_model.training_model.fit(
            x=[X_train, Y_train],
            y=Y_train,
            batch_size=batch_size,
            epochs=rp["epochs"],
            verbose=1,
            shuffle=True,
            validation_data=([X_val, Y_val], Y_val),
        )

    # --- 2) EVM Training rounds ---
    training_rounds = [{"learning_rate": evm_learning_rate, "epochs": evm_num_epochs}]

    evm_model = AppendixEVM(
        bulk_params=bulk_model.get_parameters(),
        tanh_lo=-0.5,
        tanh_hi=0.6,
        param_threshold=0.99,
    )

    X_train = np.asarray(X_train, dtype=np.float64).reshape(-1, 1)
    Y_train = np.asarray(Y_train, dtype=np.float64).reshape(-1, 1)


    training_rounds = [
        {"learning_rate": evm_learning_rate, "epochs": evm_num_epochs},
    ]

    for i, rp in enumerate(training_rounds, 1):
        print(f"Training session {i}/{len(training_rounds)} with {rp}")

        # Compile & fit
        evm_model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=evm_model.loss,
            run_eagerly=True,   # optional for clearer traces; turn off later
        )

        feed = {
            evm_model.training_model.inputs[0].name: X_train,
            evm_model.training_model.inputs[1].name: Y_train,
        }
        evm_model.training_model.fit(
            x=feed,
            y=Y_train,
            batch_size=batch_size,
            epochs=rp["epochs"],
            verbose=1,
            shuffle=True,
            validation_data=([X_val, Y_val], Y_val),
        )

    return (evm_model, bulk_model, y_mean, y_std, history)


from pr3d.de import ConditionalGaussianMM
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Training helper definition
def train_mlp_gmm_model(
    x_dim,
    num_centers,
    hidden_sizes,
    df,
    num_samples,
    batch_size,
    num_epochs,
    learning_rate,
    standardize,
    val_fraction=0.2,
):
    # 1) Clean & align data: keep only rows with all required columns present
    cols = ["packet_delay_ms"] + list(x_dim)
    work = (df[cols]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .copy())

    # 2) Targets (Y) and optional standardization
    Y = work["packet_delay_ms"].to_numpy(np.float64).reshape(-1, 1)
    if standardize:
        y_mean = float(Y.mean())
        y_std  = float(Y.std() + 1e-8)
    else:
        y_mean = 0.0
        y_std  = 1.0
    Yz = ((Y - y_mean) / y_std).astype(np.float64)

    # 3) Feature dict (each (N,1), names must match x_dim)
    X = {k: work[k].to_numpy(np.float64).reshape(-1, 1) for k in x_dim}

    # 4) Subsample for training
    N = len(Yz)
    if num_samples > N:
        raise ValueError(f"num_samples ({num_samples}) > available samples ({N}).")
    idx = np.random.choice(N, size=num_samples, replace=False)
    X_sub = {k: v[idx] for k, v in X.items()}
    Y_sub = Yz[idx]

    # 5) Train/val split (split indices, then slice dict consistently)
    idx_train, idx_val = train_test_split(
        np.arange(len(Y_sub)), test_size=val_fraction, random_state=42
    )
    X_train = {k: v[idx_train] for k, v in X_sub.items()}
    X_val   = {k: v[idx_val]   for k, v in X_sub.items()}
    Y_train = Y_sub[idx_train]
    Y_val   = Y_sub[idx_val]

    # 6) Build model
    model = ConditionalGaussianMM(
        x_dim=x_dim,
        centers=num_centers,
        hidden_sizes=hidden_sizes,
        dtype="float64",
    )

    # (Optional) sanity check: input names
    # print("Inputs expected:", [t.name for t in model.training_model.inputs])

    # 7) Compile & train (feed dict + 'y_input')
    model.training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=model.loss,  # mean NLL in your class
    )

    train_feed = {**X_train, "y_input": Y_train}
    val_feed   = ({**X_val,   "y_input": Y_val}, Y_val)

    history = model.training_model.fit(
        x=train_feed,
        y=Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=True,
        verbose=1,
        validation_data=val_feed,
    )

    return model, y_mean, y_std, history


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# evaluation helper definition
def plot_ccdf_cond_pdf_with_model(
    df: pd.DataFrame,
    model,
    max_x: float = 400.0,
    y_mean: float | None = None,
    y_std: float | None = None,
    condition: dict | None = None,      # e.g. {"mretx": 0} or {"len": (700,800)}
    features: list[str] | None = None,  # defaults to model's input names
    num_points: int = 800,
    pdf_bins: int = 200,
    max_cond_samples: int = 4000,
    batch_size: int = 2048,
    y_block: int = 64,                   # how many y-grid points per prediction block
    use_kde: bool = False,
    label: str | None = None,
):
    """
    Plot empirical vs. model CCDF/PDF for a CONDITIONAL mixture model.

    - Filters df by `condition` (exact match or (lo,hi) range).
    - Averages p(y|x) over the filtered X to get a conditional-marginal curve.
    """

    # ---------- pick feature names from model if not given ----------
    if features is None:
        try:
            features = list(model.core_model.input_slices.keys())
        except Exception:
            raise ValueError("Could not infer feature names from model; please pass `features=`.")

    # ---------- filter rows ----------
    data = df.copy()
    if condition:
        mask = np.ones(len(data), dtype=bool)
        for k, v in condition.items():
            if callable(v):
                mask &= data[k].apply(v).astype(bool).to_numpy()
            elif isinstance(v, tuple) and len(v) == 2:
                lo, hi = v
                mask &= (data[k] >= lo) & (data[k] <= hi)
            else:
                mask &= (data[k] == v)
        data = data[mask]

    # ---------- keep only rows with all required columns ----------
    needed_cols = ["packet_delay_ms"] + features
    data = data[needed_cols].apply(pd.to_numeric, errors="coerce").dropna()

    if data.empty:
        raise ValueError("No valid rows after filtering/cleaning.")

    # ---------- empirical CCDF/PDF ----------
    delays = data["packet_delay_ms"].to_numpy(float)
    delays = delays[(delays >= 0) & (delays <= max_x)]
    delays.sort()
    n = delays.size
    if n == 0:
        raise ValueError("No valid delay samples in the specified range.")
    emp_ccdf = 1.0 - (np.arange(1, n + 1) / n)

    # empirical PDF (KDE if asked; else histogram density)
    emp_x_pdf, emp_pdf = None, None
    if use_kde:
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(delays)
            emp_x_pdf = np.linspace(0, max_x, num_points)
            emp_pdf   = kde(emp_x_pdf)
        except Exception:
            pass
    if emp_pdf is None:
        counts, edges = np.histogram(delays, bins=pdf_bins, range=(0, max_x), density=True)
        emp_x_pdf = 0.5 * (edges[:-1] + edges[1:])
        emp_pdf   = counts

    # ---------- build X dict for model ----------
    # sample at most max_cond_samples rows for speed
    if len(data) > max_cond_samples:
        sel_idx = np.random.choice(len(data), size=max_cond_samples, replace=False)
        data_sel = data.iloc[sel_idx]
    else:
        data_sel = data

    # each feature as (M,1) float64
    X_sub = {k: data_sel[k].to_numpy(np.float64).reshape(-1, 1) for k in features}
    M = len(next(iter(X_sub.values())))
    if M == 0:
        raise ValueError("No feature rows available after sampling.")

    # ---------- y-grid (and standardization) ----------
    grid_y = np.linspace(0.0, max_x, num_points)
    if (y_mean is not None) and (y_std is not None):
        z_grid = (grid_y - y_mean) / (y_std + 1e-12)
        jac = 1.0 / (y_std + 1e-12)     # d z / d y
    else:
        z_grid = grid_y
        jac = 1.0

    # ---------- predict model curves by averaging over X ----------
    pdf_pred = np.empty_like(grid_y)
    ccdf_pred = np.empty_like(grid_y)

    # Process y in blocks for efficiency
    for s in range(0, num_points, y_block):
        e = min(s + y_block, num_points)
        k = e - s
        z_blk = z_grid[s:e]                                # (k,)

        # repeat features k times to align with y repeated per x
        X_blk = {name: np.tile(arr, (k, 1)) for name, arr in X_sub.items()}  # ((k*M), 1)
        y_rep = np.repeat(z_blk, repeats=M).reshape(-1, 1)                    # ((k*M), 1)

        feed = {**X_blk, "y_input": y_rep}
        pdf_vec, _, ecdf_vec = model.prob_pred_model.predict(feed, batch_size=batch_size, verbose=0)

        pv = np.squeeze(pdf_vec).reshape(k, M)           # (k, M)
        ev = np.squeeze(ecdf_vec).reshape(k, M)          # (k, M)  (ecdf already expanded to (.,1) in model)

        pdf_pred[s:e]  = pv.mean(axis=1) * jac
        ccdf_pred[s:e] = 1.0 - ev.mean(axis=1)

    # Decide label
    if label is None:
        if condition:
            cond_text = ", ".join(
                [f"{k}={v[0]}..{v[1]}" if isinstance(v, tuple) else f"{k}={v}" for k, v in condition.items()]
            )
            label = f"Model (avg over X | {cond_text})"
        else:
            label = "Model (avg over X)"

    # ===================== PLOTS =====================

    # CCDF
    plt.figure(figsize=(6, 4))
    plt.step(delays, emp_ccdf, where='post', label='Empirical CCDF')
    plt.plot(grid_y, ccdf_pred, label=label)
    plt.yscale('log')
    plt.xlim(0, max_x)
    plt.ylim(1e-5, 1)
    plt.xlabel('Packet delay (ms)')
    plt.ylabel('CCDF  (P[Delay > x])')
    plt.title('Packet Delay CCDF: Conditional Data vs Model')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PDF
    plt.figure(figsize=(6, 4))
    if len(emp_x_pdf) == len(grid_y):
        plt.plot(emp_x_pdf, emp_pdf, label='Empirical PDF')
    else:
        plt.step(emp_x_pdf, emp_pdf, where='mid', label='Empirical PDF')
    plt.plot(grid_y, pdf_pred, label=label)
    plt.xlim(0, max_x)
    plt.xlabel('Packet delay (ms)')
    plt.ylabel('PDF')
    plt.title('Packet Delay PDF: Conditional Data vs Model')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "grid_y": grid_y,
        "pdf_pred": pdf_pred,
        "ccdf_pred": ccdf_pred,
        "emp_x_pdf": emp_x_pdf,
        "emp_pdf": emp_pdf,
        "emp_delays": delays,
        "emp_ccdf": emp_ccdf,
        "num_X_used": M,
    }
