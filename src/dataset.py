import math
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Fixed categorical vocabularies
TOUR_VOCAB = {"ATP": 0, "WTA": 1}
SURFACE_VOCAB = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3, "Unknown": 4}
FORMAT_VOCAB = {"best_of_3": 0, "best_of_5": 1, "unknown": 2}


def build_draw_vocab(train_df: pd.DataFrame) -> dict[str, int]:
    """Fit draw vocabulary strictly on training data with an 'Unknown' fallback."""
    draws = sorted(set(train_df["draw"].dropna().astype(str)) - {"Unknown"})
    vocab = {label: idx for idx, label in enumerate(draws)}
    vocab["Unknown"] = len(vocab)
    return vocab


def infer_match_format(tour: str, draw: str, surface: str) -> str:
    """Infer prematch format (best_of_3, best_of_5, or unknown) from match metadata."""
    if surface == "Unknown":  # Davis/Fed Cup team ties
        return "unknown"
    draw_str = str(draw).lower()
    if "qual" in draw_str or draw_str == "unknown":
        return "unknown"
    # Men's Grand Slam main draws (draw size 128) are best-of-5
    if tour == "ATP" and draw_str == "128":
        return "best_of_5"
    return "best_of_3"


def encode_point_score(pts_s: int, pts_r: int) -> tuple[int, int]:
    """Map raw point counts to 5 categories: 0->0, 15->1, 30->2, 40->3, AD->4."""
    if pts_s < 3 and pts_r < 3:
        return pts_s, pts_r
    if pts_s == pts_r:  # Deuce (40-40, 4-4, 5-5, etc.)
        return 3, 3
    if pts_s > pts_r:   # Server Advantage (AD-40)
        return 4, 3
    return 3, 4         # Returner Advantage (40-AD)


class TennisMatchDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        draw_vocab: dict[str, int],
        lam: float = 20.0,
    ):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.draw_vocab = draw_vocab
        self.num_draws = len(draw_vocab)
        self.lam = lam

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        points = row["points"]
        T = len(points)

        # Match-level categorical encodings (constant across all T points)
        tour_idx = TOUR_VOCAB[row["tour"]]
        surf_idx = SURFACE_VOCAB.get(row["surface"], SURFACE_VOCAB["Unknown"])
        draw_idx = self.draw_vocab.get(str(row["draw"]), self.draw_vocab["Unknown"])
        fmt_str = infer_match_format(row["tour"], row["draw"], row["surface"])
        fmt_idx = FORMAT_VOCAB[fmt_str]

        tour_oh = F.one_hot(torch.tensor(tour_idx), num_classes=len(TOUR_VOCAB)).float()
        surf_oh = F.one_hot(torch.tensor(surf_idx), num_classes=len(SURFACE_VOCAB)).float()
        draw_oh = F.one_hot(torch.tensor(draw_idx), num_classes=self.num_draws).float()
        fmt_oh = F.one_hot(torch.tensor(fmt_idx), num_classes=len(FORMAT_VOCAB)).float()
        match_meta_oh = torch.cat([tour_oh, draw_oh, surf_oh, fmt_oh], dim=0)

        a_serve_prior = float(row["a_serve_prior"])
        a_return_prior = float(row["a_return_prior"])
        b_serve_prior = float(row["b_serve_prior"])
        b_return_prior = float(row["b_return_prior"])

        # Allocate per-step tensors
        x_seq = torch.zeros((T, 4), dtype=torch.float32)       # Previous point outcome (x_0 = 0)
        s_list = []                                            # Current score & context s_{t+1}
        z_seq = torch.zeros((T, 2), dtype=torch.float32)       # [server_prior, returner_prior]
        c_seq = torch.zeros((T, 5), dtype=torch.float32)       # [log(1 + n_tilde), logit(q_t)]
        y_seq = torch.zeros(T, dtype=torch.float32)            # Target: 1 if next server wins
        loss_mask = torch.zeros(T, dtype=torch.float32)        # 1 for non-tiebreak points
        mask_30_30 = torch.zeros(T, dtype=torch.bool)          # True for 30-30 non-tiebreak points

        # Running order-free point counts before step t: [n_AA, n_AB, n_BA, n_BB]
        n_AA, n_AB, n_BA, n_BB = 0, 0, 0, 0

        for t, pt in enumerate(points):
            server = pt["server"]   # 0 = Player A, 1 = Player B
            winner = pt["winner"]   # 0 = Player A, 1 = Player B
            is_tb = bool(pt["is_tiebreak"])

            # 1. History input x_t (outcome of point t-1; stays all-zeros at t = 0)
            if t > 0:
                prev = points[t - 1]
                outcome_cat = prev["server"] * 2 + prev["winner"]  # 0:AA, 1:AB, 2:BA, 3:BB
                x_seq[t, outcome_cat] = 1.0

            # 2. Current score & context s_{t+1} (from next server's perspective)
            pts_s_idx, pts_r_idx = encode_point_score(pt["pts_s"], pt["pts_r"])
            games_s_idx = min(pt["games_s"], 7)
            games_r_idx = min(pt["games_r"], 7)
            sets_s_idx = min(pt["sets_s"], 2)
            sets_r_idx = min(pt["sets_r"], 2)
            game_diff = float(pt["games_s"] - pt["games_r"])

            step_oh = torch.cat([
                F.one_hot(torch.tensor(pts_s_idx), num_classes=5).float(),
                F.one_hot(torch.tensor(pts_r_idx), num_classes=5).float(),
                F.one_hot(torch.tensor(games_s_idx), num_classes=8).float(),
                F.one_hot(torch.tensor(games_r_idx), num_classes=8).float(),
                F.one_hot(torch.tensor(sets_s_idx), num_classes=3).float(),
                F.one_hot(torch.tensor(sets_r_idx), num_classes=3).float(),
                F.one_hot(torch.tensor(server), num_classes=2).float(),
                match_meta_oh,
                torch.tensor([game_diff], dtype=torch.float32),
            ], dim=0)
            s_list.append(step_oh)

            # 3. Frozen priors z & 4. Reordered cumulative counts n_tilde_t
            if server == 0:  # Player A serves next
                srv_prior, ret_prior = a_serve_prior, b_return_prior
                n_SS, n_SR, n_RS, n_RR = n_AA, n_AB, n_BA, n_BB
            else:            # Player B serves next
                srv_prior, ret_prior = b_serve_prior, a_return_prior
                n_SS, n_SR, n_RS, n_RR = n_BB, n_BA, n_AB, n_AA

            z_seq[t, 0] = srv_prior
            z_seq[t, 1] = ret_prior

            # Empirical Bayes in-match serve win rate q_t
            r_S = 1.0 / (1.0 + math.exp(-srv_prior))
            q_t = (n_SS + self.lam * r_S) / (n_SS + n_SR + self.lam)
            logit_q_t = math.log(q_t / (1.0 - q_t))

            c_seq[t, 0] = math.log1p(n_SS)
            c_seq[t, 1] = math.log1p(n_SR)
            c_seq[t, 2] = math.log1p(n_RS)
            c_seq[t, 3] = math.log1p(n_RR)
            c_seq[t, 4] = logit_q_t

            # 5. Target y_{t+1} and masks
            y_seq[t] = 1.0 if (winner == server) else 0.0
            if not is_tb:
                loss_mask[t] = 1.0
                if pt["pts_s"] == 2 and pt["pts_r"] == 2:
                    mask_30_30[t] = True

            # Update running counts AFTER recording step t features (no leakage)
            if server == 0 and winner == 0:
                n_AA += 1
            elif server == 0 and winner == 1:
                n_AB += 1
            elif server == 1 and winner == 0:
                n_BA += 1
            else:
                n_BB += 1

        s_seq = torch.stack(s_list, dim=0)

        return {
            "x": x_seq,                  # (T, 4)
            "s": s_seq,                  # (T, D_s)
            "z": z_seq,                  # (T, 2)
            "c": c_seq,                  # (T, 5)
            "y": y_seq,                  # (T,)
            "loss_mask": loss_mask,      # (T,)
            "mask_30_30": mask_30_30,    # (T,)
            "length": T,
        }


def collate_matches(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length match sequences in a batch to max_T."""
    batch_size = len(batch)
    max_T = max(item["length"] for item in batch)
    D_s = batch[0]["s"].shape[1]

    x_pad = torch.zeros((batch_size, max_T, 4), dtype=torch.float32)
    s_pad = torch.zeros((batch_size, max_T, D_s), dtype=torch.float32)
    z_pad = torch.zeros((batch_size, max_T, 2), dtype=torch.float32)
    c_pad = torch.zeros((batch_size, max_T, 5), dtype=torch.float32)
    y_pad = torch.zeros((batch_size, max_T), dtype=torch.float32)
    mask_pad = torch.zeros((batch_size, max_T), dtype=torch.float32)
    mask_30_30_pad = torch.zeros((batch_size, max_T), dtype=torch.bool)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["length"]
        x_pad[i, :T] = item["x"]
        s_pad[i, :T] = item["s"]
        z_pad[i, :T] = item["z"]
        c_pad[i, :T] = item["c"]
        y_pad[i, :T] = item["y"]
        mask_pad[i, :T] = item["loss_mask"]
        mask_30_30_pad[i, :T] = item["mask_30_30"]

    return {
        "x": x_pad,
        "s": s_pad,
        "z": z_pad,
        "c": c_pad,
        "y": y_pad,
        "loss_mask": mask_pad,
        "mask_30_30": mask_30_30_pad,
        "lengths": lengths,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    train_path = root / "data/processed/train.parquet"

    train_df = pd.read_parquet(train_path)
    draw_vocab = build_draw_vocab(train_df)

    ds = TennisMatchDataset(train_path, draw_vocab=draw_vocab, lam=20.0)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_matches)

    batch = next(iter(loader))
    for k, v in batch.items():
        print(f"{k:12s}: {tuple(v.shape)} ({v.dtype})")