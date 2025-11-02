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

data = torch.load("dataset.pt")

x = data['points']
print('x',x.shape)
coefs = data['a']
print('coefs',coefs.shape)
epsilons = data['epsilons']
print('epsilons',epsilons.shape)
solutions = data['uh']
print('solutions',solutions.shape)

device = torch.device("cpu")
torch.set_default_device(device)

class NOMAD(nn.Module):
    def __init__(self, activation_branch, activation_trunk, branch_input_dim:int, trunk_input_dim:int, 
                 branch_hidden_dims:list, trunk_hidden_dims:list, num_basis:int, decoder_hidden_dims:list,
                 use_trunk_net=True):
        super(NOMAD, self).__init__()

        self.use_trunk_net = use_trunk_net

        # Branch: coeffs → β
        self.branch_net = DeepNet(activation_branch, branch_input_dim, branch_hidden_dims, num_basis)

        # Trunk: x → embedding(x)
        if self.use_trunk_net:
            self.trunk_net = DeepNet(activation_trunk, trunk_input_dim, trunk_hidden_dims, num_basis)
        else:
            self.trunk_net = nn.Identity()
            
        # Decoder: (β, trunk(x), epsilon) → u(x)
        decoder_input_dim = num_basis + num_basis + 1  # β + x_embed + epsilon
        decoder_layers = []
        input_dim = decoder_input_dim
        for h in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(input_dim, h))
            decoder_layers.append(nn.ReLU())
            input_dim = h
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, coefs, x, epsilons):
        B = coefs.shape[0]      # batch size
        N = x.shape[0]          # number of spatial points

        beta = self.branch_net(coefs)         # [B, num_basis]
        
        x = x.unsqueeze(-1)        # da [N] → [N, 1]
        x = x.unsqueeze(0)         # da [N, 1] → [1, N, 1]
        x = x.expand(B, N, 1)      # da [1, N, 1] → [B, N, 1]
        x_raw = x.clone()  # [B, N, 1], prima dell'embed
        
        if self.use_trunk_net:
            x_embed = self.trunk_net(x)       # [B, N, num_basis]
        else:
            x_embed = x.repeat(1, 1, beta.shape[1])  # [B, N, num_basis]
        
        beta = beta.unsqueeze(1).expand(-1, N, -1)       # [B, N, num_basis]
        eps = epsilons.view(B, 1, 1).expand(-1, N, 1)     # [B, N, 1]

        decoder_input = torch.cat([beta, x_embed, eps], dim=-1)  # [B, N, total_input]
        output = self.decoder(decoder_input).squeeze(-1)         # [B, N]
        
        output = output * x_raw.squeeze(-1) * (1 - x_raw.squeeze(-1))

        return output


class BalancedLoss(nn.Module):
    def __init__(self, x):
        super(BalancedLoss, self).__init__()
        self.x = x
        self.dx = x[1:] - x[:-1]  # Differenze finite

    def forward(self, y_pred, y_true, epsilons):
        mse_u_per_sample = torch.mean((y_pred - y_true) ** 2, dim=1)

        # Derivate numeriche con differenze finite
        dy_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / self.dx
        dy_true = (y_true[:, 1:] - y_true[:, :-1]) / self.dx
        mse_du_per_sample = torch.mean((dy_pred - dy_true) ** 2, dim=1)

        # Balanced loss per sample
        loss_per_sample = epsilons * mse_du_per_sample + mse_u_per_sample
        return loss_per_sample.mean()


# Architettura
activation_b = nn.ReLU()
activation_t = nn.ReLU()
branch_input_dim = coefs.shape[1]  
branch_hidden_dims = [50, 50]
beta_dim = 32

decoder_input_dim = beta_dim + epsilons.shape[-1]  
decoder_hidden_dims = [64, 64]

model = NOMAD(
    activation_branch=activation_b,
    activation_trunk=activation_t,
    branch_input_dim=50,
    trunk_input_dim=1,
    branch_hidden_dims=[50, 50],
    trunk_hidden_dims=[50, 50, 70],
    num_basis=20,
    decoder_hidden_dims=[100, 100]
)


model


# Train the model
num_epochs = int(5000)
learning_rate = 5e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1250, gamma=0.5)

# Choose a criterion
criterion = BalancedLoss(x)

def train_nomad(model, x, coefs, epsilons, solutions, num_epochs, loss_values, scriterion):
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(coefs.to(device), x.to(device), epsilons.to(device))  # [B, N]
        
        # Compute loss
        loss = criterion(output, solutions.to(device), epsilons.to(device))  
        # Backward pass
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Store loss
        loss_values.append(loss.item())

        # Logging
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {loss.item():.4e}")
    
    return loss_values

loss_values = []
start_time = time.time()  
loss_values = train_nomad(model, x, coefs, epsilons, solutions, num_epochs, loss_values, criterion)
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds")

# Grafici interessanti
plt.semilogy(range(len(loss_values)), loss_values)
plt.grid()

# training set
def evaluate_nomad(model, x, coefs, epsilons, solutions, criterion):
    model.eval()
    with torch.no_grad():
        u_pred = model(coefs.to(device), x.to(device), epsilons.to(device))
        loss = criterion(u_pred, solutions.to(device),epsilons.to(device))
        print(f"Test Loss: {loss.item():.4e}")
    return u_pred


# Plot results
predicted_solutions = evaluate_nomad(model, x, coefs, epsilons, solutions, criterion)
print(solutions.shape)
plot_results(x.detach(), solutions, predicted_solutions.detach(),'Training set', selected_indices=[127,5,62])

# test set
data_test = torch.load("testset.pt")
x_test = data_test['points']
coefs_test = data_test['a']
solutions_test = data_test['uh']
epsilon_test = data_test['epsilons']
# see the dimension of the data
print("Dimension of x: ", x_test.shape)
print("Dimension of coefs: ", coefs_test.shape)
print("Dimension of solutions: ", solutions_test.shape)


predicted_solutions_test = evaluate_nomad(model, x_test, coefs_test, epsilon_test, solutions_test, criterion)
import torch.nn.functional as F

mse = F.mse_loss(predicted_solutions_test, solutions_test)
print(mse)
plot_results(x_test.detach(), solutions_test, predicted_solutions_test.detach(), 'Test set', selected_indices=[2, 4,1])


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


