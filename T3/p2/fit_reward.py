import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, output_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return x

    def get_difference_logit(self, p_A, p_B):
        return self.forward(p_A) - self.forward(p_B)




# división de dataset 70 train 15 val 15 test
def split_dataset(dataset):
    N = len(dataset)

    idx = np.random.permutation(N)

    n_train = int(0.7*N)
    n_val = int(0.15*N)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


### PERFILES
FEATURES = [
    "anthocyanin",
    "tannin",
    "catechin",
    "color_intensity",
    "glycerol",
    "volatile_acidity",
    "ethanol_pct",
    "residual_sugar",
    "fermentation_h"
]

def profile_to_vec(profile):
    return np.array(
        [profile[k] for k in FEATURES], dtype=np.float32
    )


def main():
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score
    from scripts.quality_functions import get_wine_quality_score

    asd = input("Entrenar modelo o cargar modelo? ([1] train  /  [2] load): ")
    modo = "train" if asd == "1" else "load"

    ##### CARGAR DATASET
    dataset = pickle.load(open(r"p2\data\comparisons.pkl", "rb"))

    X_A = []
    X_B = []
    Y = []

    for sample in dataset:
        X_A.append(profile_to_vec(
            sample["profile_A"]
            )
        )

        X_B.append(profile_to_vec(
            sample["profile_B"]
            )
        )

        Y.append(1 if sample["winner"] == "A" else 0)

    X_A = np.array(X_A)
    X_B = np.array(X_B)
    Y = np.array(Y)

    ## splitteamos ahora a train, val y test
    dataset_splits = split_dataset(dataset)

    train_idx = dataset_splits["train"]
    val_idx = dataset_splits["val"]
    test_idx = dataset_splits["test"]

    all_profiles = np.concatenate(
        [X_A[train_idx], X_B[train_idx]],
        axis=0
    )

    ## NORMALIZAR
    mu = all_profiles.mean(axis=0)
    std = all_profiles.std(axis=0) + 1e-8

    X_A = (X_A - mu) / std
    X_B = (X_B - mu) / std

    ## CONJUNTOS train/val/test
    XA_train    = torch.tensor(X_A[train_idx], dtype=torch.float32)
    XB_train    = torch.tensor(X_B[train_idx], dtype=torch.float32)
    Y_train     = torch.tensor(Y[train_idx], dtype=torch.float32)

    XA_val      = torch.tensor(X_A[val_idx], dtype=torch.float32)
    XB_val      = torch.tensor(X_B[val_idx], dtype=torch.float32)
    Y_val       = torch.tensor(Y[val_idx], dtype=torch.float32)

    XA_test     = torch.tensor(X_A[test_idx], dtype=torch.float32)
    XB_test     = torch.tensor(X_B[test_idx], dtype=torch.float32)
    Y_test      = torch.tensor(Y[test_idx], dtype=torch.float32)

    ### crear model y entrenar
    model = RewardNet(
        input_dim=len(FEATURES),
        output_dim=1
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ### loop
    epochs = 100

    train_losses = []
    val_losses = []

    best_val_loss = np.inf

    if modo == "train":
        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f"Epoch {epoch} / {epochs}   |   Train loss: {train_losses[-1] if train_losses else 'N/A'}   |   Val loss: {val_losses[-1] if val_losses else 'N/A'}")

            model.train()

            logits = model.get_difference_logit(
                XA_train.to(device),
                XB_train.to(device)
            ).squeeze()

            loss = criterion(
                logits,
                Y_train.to(device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # eval
            model.eval()

            with torch.no_grad():
                val_logits = model.get_difference_logit(
                    XA_val.to(device),
                    XB_val.to(device)
                ).squeeze()

                val_loss = criterion(
                    val_logits,
                    Y_val.to(device)
                )

                val_losses.append(val_loss.item())

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), "best_reward_model.pt")

    ### cargar mejor modelo
    model.load_state_dict(
        torch.load("best_reward_model.pt", map_location=device)
    )

    ## accuracy
    model.eval()

    with torch.no_grad():
        logits = model.get_difference_logit(
            XA_test.to(device),
            XB_test.to(device)
        ).squeeze()

        preds = (logits > 0).cpu().numpy()

        acc = np.mean(preds == Y_test.numpy())

        print(f"Test accuracy: {acc:.4f}")

    ### MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(Y_test.numpy(), preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()
    plt.savefig("results/p2/confusion_matrix.png", dpi=300)

    # R^2
    unique_profiles = []
    seen = set()

    for sample in dataset:
        for profile in (sample["profile_A"], sample["profile_B"]):
            key = tuple(profile[k] for k in FEATURES)
            if key not in seen:
                seen.add(key)
                unique_profiles.append(profile)

    true_scores = []
    pred_scores = []

    model.eval()
    with torch.no_grad():
        for profile in unique_profiles:
            x = profile_to_vec(profile)
            x = (x - mu) / std
            x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

            true_scores.append(get_wine_quality_score(profile))
            pred_scores.append(model(x).squeeze().item())

    pred_arr = np.array(pred_scores)
    true_arr = np.array(true_scores)

    # ajuste a escala de 0 a 8
    true_min, true_max = true_arr.min(), true_arr.max()
    pred_rescaled = (pred_arr - pred_arr.min()) / (pred_arr.max() - pred_arr.min())
    pred_rescaled = pred_rescaled * (true_max - true_min) + true_min

    r2 = r2_score(true_arr, pred_rescaled)
    print(f"R^2: {r2:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(true_arr, pred_rescaled, alpha=0.6) # pred_rescaled
    lims = [
        min(min(true_arr), min(pred_rescaled)),
        max(max(true_arr), max(pred_rescaled)),
    ]

    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("true_scores")
    plt.ylabel("pred_scores")
    plt.title("True vs Predicted reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/p2/reward_fit.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()