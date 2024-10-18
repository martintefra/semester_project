import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import get_laplacian
from torch.nn import functional as F
from torch_geometric.datasets import LRGBDataset


def compute_dirichlet_energy(signal, edge_index, device):
    # Function taken from https://github.com/JThh/Many-Body-MPNN/tree/main
    
    # compute the normalized laplacian: delta = I - D^(-1/2) * A * D^(-1/2)
    lap_indices, lap_values = get_laplacian(edge_index, normalization="sym", dtype=torch.float)

    laplacian_sparse = torch.sparse_coo_tensor(lap_indices, lap_values).to(device)
    
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)  # Make it (N, 1) if it's just (N,)
        
    signal = signal.to(device)

    # Compute Dirichlet energy = signal^T * L * signal
    torch.matmul(signal.t(), torch.matmul(laplacian_sparse, signal))
    energy = torch.trace(torch.matmul(signal.t(), torch.sparse.mm(laplacian_sparse, signal)))

    return energy.item()


def train(model, train_loader, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        target = torch.argmax(data.y, dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        target = torch.argmax(data.y, dim=1)
        correct += int((pred == target).sum()) 
    return correct / len(loader.dataset)


def load_dataset(dataset_name):
    if dataset_name == "peptides":
        dataset = LRGBDataset(root="./data/LRGB", name="Peptides-func")
        return dataset
    else:
        raise ValueError("Dataset not supported")
    
    
def plot_results(mean_arr, ci_arr, model_type, dataset_name):
    epochs = np.arange(1, len(mean_arr) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_arr, color="blue", marker="o")
    plt.fill_between(epochs, mean_arr - ci_arr, mean_arr + ci_arr, color="blue", alpha=0.3)


    plt.title(f'Dirichlet Energy - {dataset_name} - {model_type}')
    plt.xlabel('Epochs')
    plt.ylabel('Dirichlet Energy')
    plt.grid(True)

    # Show the plots
    plt.savefig(f'./figures/dirichlet_energy_{dataset_name}_{model_type}.png')
    plt.show()


# Function to run one experiment
def run_experiment(train_loader, val_loader, train_loader_for_energy, model, device, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dirichlet_energies = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, device)  # Assuming you have this defined elsewhere
        train_acc = test(model, train_loader, device)
        val_acc = test(model, val_loader, device)
        
        # Store the accuracies
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch: {epoch}, Train: {train_acc:.4f}, Val: {val_acc:.4f}')
        
        # Compute the energy for the full graph
        data = next(iter(train_loader_for_energy))
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch, use_pooling=False)
        energy = compute_dirichlet_energy(out.detach(), data.edge_index, device)
        
        # Store the Dirichlet energy
        dirichlet_energies.append(energy)
        print(f'Energy: {energy:.4f}')
    
    return train_accuracies, val_accuracies, dirichlet_energies


def compute_confidence_intervals(runs, alpha=0.95):
    mean = np.mean(runs, axis=0)
    std_err = np.std(runs, axis=0) / np.sqrt(len(runs))
    ci = std_err * 1.96  # For 95% confidence interval
    return mean, ci