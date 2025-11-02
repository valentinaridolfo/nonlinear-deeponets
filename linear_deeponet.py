# tutto nel branch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from DNN import DeepNet 
from utilities import plot_data, plot_results, LprelLoss
import time

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_default_dtype(torch.float64)

# Carica i dati
data = torch.load("dataset.pt")

x = data['points']
coefs = data['a']
epsilons = data['epsilons']
solutions = data['uh']

print(epsilons)

device = torch.device("cpu")
torch.set_default_device(device)

class DeepONet(nn.Module):
    def __init__(self, activation_branch, activation_trunk, branch_input_dim:int, trunk_input_dim:int, 
                 branch_hidden_dims:list, trunk_hidden_dims:list, num_basis:int, bc:bool=False):
        """
        DeepONet class. This class defines the architecture of the DeepONet.
        The DeepONet is composed of two subnetworks: the branch network and the trunk network.
        
        Args:
        activation: activation function
        branch_input_dim: input dimension of the branch network
        trunk_input_dim: input dimension of the trunk network
        branch_hidden_dims: list of the hidden dimensions of the branch network
        trunk_hidden_dims: list of the hidden dimensions of the trunk network
        num_basis: number of basis functions
        bc: boolean, if True, we multiply the output of the trunk network by x*(1-x) to satisfy the boundary condition
        """
        super(DeepONet, self).__init__()

        # The output dimension of the trunk network must be equal to the number of basis functions.
        assert trunk_hidden_dims[-1] == num_basis 

        self.activation_branch = activation_branch
        self.activation_trunk = activation_trunk
        self.bc = bc
        
        # Aggiorniamo il branch_net per prendere sia coefs che epsilons
        self.branch_net = DeepNet(activation_branch, branch_input_dim + 1, branch_hidden_dims, num_basis)  # +1 per epsilons
        self.trunk_net = DeepNet(activation_trunk, trunk_input_dim, trunk_hidden_dims, None)        

    def forward(self, f_branch, x_trunk, epsilons):
        epsilons = epsilons.unsqueeze(-1)
        # Concatenazione coefs ed epsilons nel branch network
        branch_input = torch.cat((f_branch, epsilons), dim=-1)  # Concatenate along the last dimension
        
        branch_output = self.branch_net(branch_input)
        
        trunk_output = self.trunk_net(x_trunk)
        

        
        if self.bc:
            trunk_output = trunk_output.T * x_trunk[:] * (x_trunk[:] - 1.0)
            return branch_output @ trunk_output
        else:
            return branch_output @ trunk_output.T

    def grad_forward(self, f_branch, x_trunk, epsilons):
        x_trunk.requires_grad_(True)
        branch_output = self.branch_net(f_branch)
        trunk_output = self.trunk_net(x_trunk)
        grad_trunk = torch.zeros_like(trunk_output)
        for i in range(trunk_output.shape[1]):
            grad_trunk[:, i] = torch.autograd.grad(trunk_output[:, i].sum(), x_trunk, create_graph=True, retain_graph=True)[0]
        if self.bc:
            grad_trunk = x_trunk[:] * (x_trunk[:]-1.0) * grad_trunk.T + (2*x_trunk[:]-1.0) * trunk_output.T
            return branch_output @ grad_trunk
        else:
            return branch_output @ grad_trunk.T
        
        
class BalancedLoss(nn.Module):
    def __init__(self, x):
        super(BalancedLoss, self).__init__()
        self.x = x
        self.dx = x[1:] - x[:-1]  # differenze finite (costanti per ora)

    def forward(self, y_pred, y_true, epsilons):
        mse_u_per_sample = torch.mean((y_pred - y_true) ** 2, dim=1)

        # Derivate numeriche con differenze finite
        dy_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.dx
        dy_true = (y_true[:, 1:] - y_true[:, :-1]) / self.dx
        mse_du_per_sample = torch.mean((dy_pred - dy_true) ** 2, dim=1)

        # Balanced loss per sample
        loss_per_sample = epsilons * mse_du_per_sample + mse_u_per_sample
        return loss_per_sample.mean()




# Parametri del modello
activation_b = nn.ReLU()
activation_t = nn.ReLU()
branch_input_dim = len(x)  # La dimensione di x è la dimensione dell'input per il branch network
trunk_input_dim = 1  # Trunk network si occupa della variabile x
branch_hidden_dims = [50, 50]
trunk_hidden_dims = [50, 50, 20]
num_basis = 20
bc = True

# Creazione del modello
model = DeepONet(activation_branch=activation_b,
                 activation_trunk=activation_t,
                 branch_input_dim=branch_input_dim, 
                 trunk_input_dim=trunk_input_dim, 
                 branch_hidden_dims=branch_hidden_dims,
                 trunk_hidden_dims=trunk_hidden_dims,
                 num_basis=num_basis,
                 bc=bc).to(device)

# Parametri di addestramento
num_epochs = 5000
learning_rate = 5e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1250, gamma=0.5)

# Scegli il criterio
criterion = BalancedLoss(x)



def train_deeponet(model, x, rhs, epsilons, solutions, num_epochs, loss_values, criterion):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(rhs.to(device), x.to(device), epsilons.to(device))
        loss = criterion(output, solutions.to(device), epsilons.to(device))  
        loss.backward()
        optimizer.step()
        loss_values.append(loss.cpu().detach().numpy())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4e}')
    return loss_values


# Inizializza la lista dei valori di perdita
loss_values = []
start_time = time.time()
loss_values = train_deeponet(model, x, coefs, epsilons, solutions, num_epochs, loss_values, criterion)
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Grafico dei valori di perdita
plt.semilogy(range(len(loss_values)), loss_values)
plt.grid()

# Funzione di valutazione
def evaluate_deeponet(model, x, coefs, epsilons, solutions, criterion):
    model.eval()
    with torch.no_grad():
        # Prediction
        u_pred = model(coefs.to(device), x.to(device), epsilons.to(device))

        # Calcolo della balanced loss con epsilons
        loss = criterion(u_pred, solutions.to(device), epsilons.to(device))

        print(f'Loss: {loss.item():.4e}')
    return u_pred


# Plot dei risultati di addestramento
predicted_solutions = evaluate_deeponet(model, x, coefs, epsilons, solutions, criterion)
plot_results(x.detach(), solutions, predicted_solutions.detach(),'Training set',  selected_indices=[127,5,62])

# Test set
data_test = torch.load("testset.pt")
x_test = data_test['points']
coefs_test = data_test['a']
solutions_test = data_test['uh']
epsilon_test = data_test['epsilons']

# Valutazione sui dati di test
predicted_solutions_test = evaluate_deeponet(model, x_test, coefs_test, epsilon_test, solutions_test, criterion)
import torch.nn.functional as F

mse = F.mse_loss(predicted_solutions_test, solutions_test)
print(mse)
plot_results(x_test.detach(), solutions_test, predicted_solutions_test.detach(),'Test set', selected_indices=[ 4,1,16])


def compute_error_metrics(preds, targets):
    # Calcolo dell'errore globale
    mse = nn.MSELoss()(preds, targets).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(preds - targets)).item()
    
    # Errore relativo punto-punto
    relative_error = torch.mean(torch.abs(preds - targets) / (torch.abs(targets) + 1e-8)).item()
    relative_error_pct = 100 * relative_error

    # Calcolo Relative L2 Error per sample
    l2_error_per_sample = torch.norm(preds - targets, dim=1) / (torch.norm(targets, dim=1) + 1e-8)
    mean_l2_error = l2_error_per_sample.mean().item()

    # Stampa
    print(f"\n Error Metrics:")
    print(f"  - MSE                : {mse:.4e}")
    print(f"  - RMSE               : {rmse:.4e}")
    print(f"  - MAE                : {mae:.4e}")
    print(f"  - Mean Rel. Error    : {relative_error_pct:.2f}%")
    print(f"  - Mean Relative L2 Error: {mean_l2_error:.4e}")

    print("\n⚡ Relative L2 Error per sample (first 10):")
    for i, err in enumerate(l2_error_per_sample[:10]):
        print(f"  Sample {i+1}: {err.item():.4e}")
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MeanRelError(%)": relative_error_pct,
        "MeanRelL2Error": mean_l2_error,
        "RelL2ErrorPerSample": l2_error_per_sample.cpu().numpy()
    }

# Training error
metrics_train = compute_error_metrics(predicted_solutions, solutions)

# Test error
metrics_test = compute_error_metrics(predicted_solutions_test, solutions_test)

def compute_l2_error_wrt_latent_dimension(model, branch_input, trunk_input, true_solutions, max_latent_dim=70):
    model.eval()
    with torch.no_grad():
        full_beta = model.branch_net(branch_input.to(device))  # [B, num_basis]
        trunk_out = model.trunk_net(trunk_input.to(device))    # [N, num_basis]

        # Se serve BC:
        #if model.bc:
        #    trunk_out = (trunk_out.T * trunk_input * (trunk_input - 1.0)).T

        errors = []
        for k in range(1, max_latent_dim + 1):
            # Prendi solo i primi k coefficienti latenti
            beta_k = full_beta.clone()
            beta_k[:, k:] = 0  # azzera quelli oltre la k-esima latente

            # Ricostruzione soluzione approssimata
            pred_k = beta_k @ trunk_out.T  # [B, N]

            # Relative L² error per sample
            rel_l2_err_k = torch.norm(pred_k - true_solutions.to(device), dim=1) / (torch.norm(true_solutions.to(device), dim=1) + 1e-8)
            mean_rel_l2_err_k = rel_l2_err_k.mean().item()
            errors.append(mean_rel_l2_err_k)

    return errors

errors_vs_latent_dim = compute_l2_error_wrt_latent_dimension(model, coefs, x, solutions, max_latent_dim=70)

import matplotlib.pyplot as plt
plt.plot(range(1, 71), errors_vs_latent_dim, marker='o')
plt.xlabel("Latent Dimension Used (k)")
plt.ylabel("Mean Relative L² Error")
plt.title("L² Error vs Latent Dimension")
plt.grid()
plt.show()
