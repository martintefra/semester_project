import torch
from torch_geometric.utils import get_laplacian
from torch.nn import functional as F

def compute_dirichlet_energy(signal, edge_index, device):
    # compute the normalized laplacian: delta = I - D^(-1/2) * A * D^(-1/2)
    lap_indices, lap_values = get_laplacian(edge_index, normalization="sym", dtype=torch.float)

    laplacian_sparse = torch.sparse_coo_tensor(lap_indices, lap_values).to(device)
    
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)  # Make it (N, 1) if it's just (N,)
        
    signal = signal.to(device)

    # Compute Dirichlet energy = signal^T * L * signal
    torch.matmul(signal.t(), torch.matmul(laplacian_sparse, signal))
    energy = torch.trace(torch.matmul(signal.t(), torch.sparse.mm(laplacian_sparse, signal)))
    
    # energy = trace()

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