
import os
import math
import random
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import WeightedRandomSampler

from dataset import build_dataset, collate_fn
from model import PepContrastNet, supervised_contrastive_loss



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == labels).sum().item())


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def mcc_from_conf(tp: int, tn: int, fp: int, fn: int) -> float:
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den != 0 else 0.0


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:

    pred = (y_score >= thr).astype(np.int64)
    tp, tn, fp, fn = confusion_from_preds(y_true, pred)
    acc = (tp + tn) / max(len(y_true), 1)
    sen = safe_div(tp, tp + fn)
    spe = safe_div(tn, tn + fp)
    prec = safe_div(tp, tp + fp)
    f1 = safe_div(2 * prec * sen, (prec + sen)) if (prec + sen) > 0 else 0.0
    mcc = mcc_from_conf(tp, tn, fp, fn)
    balacc = 0.5 * (sen + spe)
    youden = sen + spe - 1.0
    return dict(
        thr=thr,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        acc=acc,
        sen=sen,
        spe=spe,
        prec=prec,
        f1=f1,
        mcc=mcc,
        balacc=balacc,
        youden=youden,
    )



train_fasta = "train.txt"
test_fasta  = "test.txt"

train_feat  = "train_feature.txt"
test_feat   = "test_feature.txt"


def main():
    set_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] cwd    = {os.getcwd()}")


    train_ds, y_train = build_dataset(
        fasta_path=train_fasta,
        feature_txt_path=train_feat,
        max_rows=50,
    )
    test_ds, y_test = build_dataset(
        fasta_path=test_fasta,
        feature_txt_path=test_feat,
        max_rows=50,
    )

    print(f"[INFO] Train N={len(train_ds)}, Test N={len(test_ds)}")
    print(f"[INFO] Train label counts: pos={(y_train==1).sum()} / neg={(y_train==0).sum()}")
    print(f"[INFO] Test  label counts: pos={(y_test==1).sum()} / neg={(y_test==0).sum()}")

    BATCH_SIZE = 64


    USE_SAMPLER = True
    if USE_SAMPLER:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        w_pos = 1.0 / max(pos, 1)
        w_neg = 1.0 / max(neg, 1)
        sample_weights = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = Data.DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = Data.DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )

    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )


    seq_feat_dim = train_ds.input_ids.shape[2]
    mlp_in_dim   = train_ds.features.shape[1]

    model = PepContrastNet(
        seq_feat_dim=seq_feat_dim,
        mlp_in_dim=mlp_in_dim,
        seq_out_dim=64,
        feat_out_dim=64,
    ).to(device)


    p_pos = float((y_train == 1).sum() / max(len(y_train), 1))
    w_pos = 1.0 / max(2 * p_pos, 1e-8)
    w_neg = 1.0 / max(2 * (1.0 - p_pos), 1e-8)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)
    print(f"[INFO] class_weights = [neg={w_neg:.4f}, pos={w_pos:.4f}]")

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        verbose=True,
    )


    lambda_contrast = 0.2

    best_youden = -1.0
    best_model_path = "best_model.pth"


    def train_epoch() -> Tuple[float, float]:
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for enc, feat, labels in train_loader:
            enc    = enc.to(device)
            feat   = feat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, z_seq, z_feat, fused = model(enc, feat)

            ce_loss  = ce_criterion(logits, labels)
            con_loss = supervised_contrastive_loss(z_seq, labels, temperature=0.1)
            loss     = ce_loss + lambda_contrast * con_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = labels.size(0)
            loss_sum += loss.item() * bs
            correct  += count_correct(logits, labels)
            total    += bs

        return loss_sum / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def eval_epoch():

        model.eval()
        loss_sum = 0.0
        correct  = 0
        total    = 0
        all_probs  = []
        all_labels = []

        for enc, feat, labels in test_loader:
            enc    = enc.to(device)
            feat   = feat.to(device)
            labels = labels.to(device)

            logits, z_seq, z_feat, fused = model(enc, feat)
            prob = torch.softmax(logits, dim=1)[:, 1]

            pred = logits.argmax(dim=1)
            bs   = labels.size(0)
            loss = ce_criterion(logits, labels)
            loss_sum += loss.item() * bs
            correct  += int((pred == labels).sum().item())
            total    += bs

            all_probs.append(prob.detach().cpu())
            all_labels.append(labels.detach().cpu())

        acc_micro_05 = correct / max(total, 1)
        test_loss    = loss_sum / max(total, 1)
        probs        = torch.cat(all_probs,  dim=0).numpy()
        labels_np    = torch.cat(all_labels, dim=0).numpy()


        m050 = metrics_at_threshold(labels_np, probs, 0.5)
        youden_05 = m050["youden"]

        return test_loss, acc_micro_05, youden_05, m050

    max_epochs = 100
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_epoch()
        test_loss, acc05, youden_05, m050 = eval_epoch()


        scheduler.step(youden_05)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Test Loss: {test_loss:.4f} | "
            f"ACC@0.5: {acc05:.4f} | "
            f"Youden@0.5: {youden_05:.4f}"
        )


        if youden_05 > best_youden:
            best_youden = youden_05
            torch.save(model.state_dict(), best_model_path)
            print(f"→ New best model (Youden@0.5) saved -> {os.path.abspath(best_model_path)}")


        if epoch % 5 == 0 or epoch == 1:
            def fmt(m):
                return (
                    f"t={m['thr']:.3f} | ACC={m['acc']:.4f} SEN={m['sen']:.4f} "
                    f"SPE={m['spe']:.4f} MCC={m['mcc']:.4f} "
                    f"BalAcc={m['balacc']:.4f} Youden={m['youden']:.4f} | "
                    f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}"
                )
            print("    @thr=0.5 :", fmt(m050))

    print(
        f"[DONE] Max Youden@0.5 on test: {best_youden:.4f}, "
        f"best model -> {os.path.abspath(best_model_path)}"
    )


if __name__ == "__main__":
    main()
