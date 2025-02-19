import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z
    
def train_autoencoder_with_metrics(train_embeddings, val_embeddings, latent_dim, device, args):
    """
    Train Autoencoder with validation and early stopping.

    Args:
        train_embeddings (Tensor): Input embeddings for training.
        val_embeddings (Tensor): Input embeddings for validation.
        latent_dim (int): Latent dimensionality of the autoencoder.
        device (torch.device): Device to use for training (CPU or GPU).
        args (argparse.Namespace): Parsed arguments from argparse.

    Returns:
        Autoencoder: Trained autoencoder model.
        List[float]: Training reconstruction loss history per epoch.
        List[float]: Validation reconstruction loss history per epoch.
    """
    input_dim = train_embeddings.size(1)
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer_ae = optim.AdamW(ae.parameters(), lr=args.learning_rate)
    criterion_ae = nn.MSELoss()

    train_ae_loader = DataLoader(train_embeddings, batch_size=args.batch_size, shuffle=True)
    val_ae_loader = DataLoader(val_embeddings, batch_size=args.batch_size, shuffle=False)

    ae.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs_ae):
        ae.train()
        total_train_loss = 0.0
        
        optimizer_ae.zero_grad()
        
        for step, emb_batch in enumerate(train_ae_loader):
            emb_batch = emb_batch.to(device)
            reconstructed, _ = ae(emb_batch)
            train_loss = criterion_ae(reconstructed, emb_batch) / args.grad_acc_steps
            train_loss.backward()
            total_train_loss += train_loss.item()

            if (step + 1) % args.grad_acc_steps == 0 or (step + 1) == len(train_ae_loader):
                optimizer_ae.step()
                optimizer_ae.zero_grad()

        avg_train_loss = total_train_loss / len(train_ae_loader)
        train_losses.append(avg_train_loss)


        ae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for emb_batch in val_ae_loader:
                emb_batch = emb_batch.to(device)
                reconstructed, _ = ae(emb_batch)
                val_loss = criterion_ae(reconstructed, emb_batch)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_ae_loader)
        val_losses.append(avg_val_loss)

        print(f"AE(Latent={latent_dim}) Epoch {epoch+1}/{args.epochs_ae}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{args.ae_patience}")
            if patience_counter >= args.ae_patience:
                print("Early stopping triggered.")
                break

    return ae, train_losses, val_losses