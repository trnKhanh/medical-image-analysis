__all__ = ["ActiveLearningTrainer"]

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

from losses.robust_ce_loss import RobustCrossEntropyLoss
from models import UNet

from .base_trainer import BaseTrainer


class ActiveLearningTrainer(BaseTrainer):
    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_start(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_dir: str,
        initial_size: int = 200,
        budget: int = 100,
        rounds: int = 10,
        batch_size: int = 4,
        train_epochs: int = 5,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = torch.tensor(X_val).float().permute(0, 3, 1, 2).to(self.device)
        self.y_val = torch.tensor(y_val).long().to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.initial_size = initial_size
        self.budget = budget
        self.rounds = rounds
        self.batch_size = batch_size
        self.train_epochs = train_epochs

        self.current_round = 0
        self.model = None

        self._prepare_data()

    def _prepare_data(self):
        indices = np.arange(len(self.X_train))
        np.random.shuffle(indices)
        self.labeled_indices = list(indices[:self.initial_size])
        self.unlabeled_indices = list(indices[self.initial_size:])

    def on_train_start(self):
        print("Starting Active Learning...")

    def on_train_end(self):
        print("Training finished.")

    def train_step(self):
        X_labeled = self.X_train[self.labeled_indices]
        y_labeled = self.y_train[self.labeled_indices]

        self.model = UNet(n_channels=1, n_classes=np.max(self.y_train) + 1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = RobustCrossEntropyLoss()

        self.model.train()
        for epoch in tqdm(range(self.train_epochs), desc="Training Epochs"):
            for i in range(0, len(X_labeled), self.batch_size):
                images = torch.tensor(X_labeled[i:i + self.batch_size]).float().permute(0, 3, 1, 2).to(self.device)
                masks = torch.tensor(y_labeled[i:i + self.batch_size]).long().to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.save_state_dict(self.checkpoint_dir / f"unet_round_{self.current_round + 1}.pth")

    def valid_step(self):
        self.model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for i in range(0, len(self.X_val), self.batch_size):
                images = self.X_val[i:i + self.batch_size]
                masks = self.y_val[i:i + self.batch_size]

                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        dice, jaccard, precision, recall, accuracy = self.compute_metrics(y_true, y_pred)
        print(f"Validation Metrics - Dice: {dice:.4f}, Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

    def train(self):
        self.on_train_start()

        for r in range(self.current_round, self.rounds):
            self.current_round = r
            print(f"\n=== Round {r + 1} ===")

            self.train_step()
            self.valid_step()

            if not self.unlabeled_indices:
                print("No unlabeled samples left. Stopping early.")
                break

            self.query_new_samples()

        self.on_train_end()

    def query_new_samples(self):
        X_unlabeled = self.X_train[self.unlabeled_indices]
        scores = self._get_entropy_scores(X_unlabeled)
        selected_indices = np.argsort(scores)[-self.budget:]

        new_samples = [self.unlabeled_indices[i] for i in selected_indices]
        self.labeled_indices.extend(new_samples)
        self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in new_samples]

        print(f"Selected {len(new_samples)} new samples, total labeled: {len(self.labeled_indices)}")

    def compute_metrics(self, y_true, y_pred):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        dice = 2 * np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + np.sum(y_pred_flat) + 1e-6)
        jaccard = np.sum(y_true_flat * y_pred_flat) / (
            np.sum(y_true_flat) + np.sum(y_pred_flat) - np.sum(y_true_flat * y_pred_flat) + 1e-6)
        precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true_flat, y_pred_flat)

        return dice, jaccard, precision, recall, accuracy

    def run_training(self):
        self.train()
        self.perform_real_test()

    def perform_real_test(self):
        # Optional implementation placeholder
        pass

    def state_dict(self):
        return {
            "model_state": self.model.state_dict(),
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
            "current_round": self.current_round,
        }

    def save_state_dict(self, save_path: str | Path):
        torch.save(self.state_dict(), save_path)

    def load_state_dict(self, save_path: str | Path):
        state = torch.load(save_path)
        self.model = UNet(n_channels=3, n_classes=np.max(self.y_train) + 1).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.labeled_indices = state["labeled_indices"]
        self.unlabeled_indices = state["unlabeled_indices"]
        self.current_round = state["current_round"]

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        if self.model:
            self.model.to(self.device)

    def _get_entropy_scores(self, images):
        self.model.eval()
        scores = []

        with torch.no_grad():
            for i in tqdm(range(0, len(images), self.batch_size), desc="Entropy Sampling"):
                batch = torch.tensor(images[i:i + self.batch_size]).float().permute(0, 3, 1, 2).to(self.device)

                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                mean_entropy = entropy.mean(dim=(1, 2)).cpu().numpy()
                scores.extend(mean_entropy)

        return np.array(scores)
