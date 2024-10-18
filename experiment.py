import torch
import argparse
import numpy as np

from models import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from utils import plot_results, compute_confidence_intervals, run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('--model_type', type=str, default='GCN', help='Type of model to use (e.g., GCN)')
    parser.add_argument('--dataset_name', type=str, default='peptides', help='Name of the dataset to use')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs to do')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epoch for each run')
    args = parser.parse_args()
    
    device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

    ### TODO: adapt to different model types and datasets
    # # Initialize model
    # if args.model_type == 'GCN':
    #     dataset = load_dataset(args.dataset_name)
    # else:
    #     raise ValueError(f"Model type {args.model_type} not supported")


    dataset = LRGBDataset(root="./data/LRGB", name="Peptides-func")
    train_dataset = LRGBDataset(root="./data/LRGB", name="Peptides-func", split="train")
    val_dataset = LRGBDataset(root="./data/LRGB", name="Peptides-func", split="test")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_loader_for_energy = DataLoader(train_dataset, batch_size=10873, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    n_runs = args.number_runs

    train_accuracies_runs = []
    val_accuracies_runs = []
    dirichlet_energies_runs = []

    for i in range(n_runs):
        print(f'Run {i+1}/{n_runs}')
        model = GCN(dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes).to(device)
        train_acc, val_acc, dirichlet_energy = run_experiment(train_loader, val_loader, train_loader_for_energy, model, device, epochs=args.epochs)
        
        train_accuracies_runs.append(train_acc)
        val_accuracies_runs.append(val_acc)
        dirichlet_energies_runs.append(dirichlet_energy)

    # Convert lists to numpy arrays for easier handling
    train_accuracies_runs = np.array(train_accuracies_runs)
    val_accuracies_runs = np.array(val_accuracies_runs)
    dirichlet_energies_runs = np.array(dirichlet_energies_runs)

    # Compute confidence intervals
    train_acc_mean, train_acc_ci = compute_confidence_intervals(train_accuracies_runs)
    val_acc_mean, val_acc_ci = compute_confidence_intervals(val_accuracies_runs)
    energy_mean, energy_ci = compute_confidence_intervals(dirichlet_energies_runs)

    print(f"Train Accuracy Mean: {train_acc_mean[-1]:.4f} ± {train_acc_ci[-1]:.4f}")
    print(f"Val Accuracy Mean: {val_acc_mean[-1]:.4f} ± {val_acc_ci[-1]:.4f}")
    print(f"Dirichlet Energy Mean: {energy_mean[-1]:.4f} ± {energy_ci[-1]:.4f}")

    plot_results(energy_mean, energy_ci, args.model_type, args.dataset_name)
