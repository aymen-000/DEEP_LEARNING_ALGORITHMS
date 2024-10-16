import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from train import train
from test import test
from dataset import test_loader, train_loader
from vae_architacture import VAE
from visualization import visualize_tsne_vae_output, visualize_vae_latent_variable, recons

# Main function
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train and test a VAE model.")
    
    # Adding arguments
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--model_saving_path', type=str, default='vae_model.pth', help='Path to save the model')
    parser.add_argument('--vae_out_saving_path', type=str, default='vae_output.png', help='Path to save VAE output visualization')
    parser.add_argument('--latent_saving_path', type=str, default='latent_variable.png', help='Path to save latent variable visualization')
    parser.add_argument('--results_path', type=str, default='results.png', help='Path to save reconstructed images')
    
    # Parse the arguments
    args = parser.parse_args()

    # Initialize device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize TensorBoard writer (optional)
    writer = SummaryWriter(log_dir="runs/")
    # get input dims 
    input_dims = next(iter(train_loader))[0][0].shape
    # Configuration dictionary
    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "input_dims": input_dims,
        "device": device,
    }

    # Initialize the VAE model
    model = VAE(config["input_dims"], config["hidden_dim"], config["latent_dim"]).to(config["device"])

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    
    prev_updates = 0
    for epoch in range(config["num_epochs"]):
        print(f'Epoch {epoch+1}/{config["num_epochs"]}')
        
        prev_updates = train(
            model=model,
            writer=writer,
            dataloader=train_loader,
            prev_updates=prev_updates,
            optimizer=optimizer,
            device=config["device"]
        )
        
        test(model, 
             test_loader, 
             prev_updates, 
             writer=writer, 
             config=config , 
             device=config["device"])

    # Save the trained model
    torch.save(model, args.model_saving_path)
    
    # Visualize t-SNE and VAE output
    visualize_tsne_vae_output(model, test_loader, config, saving_path=args.vae_out_saving_path)
    visualize_vae_latent_variable(model, test_loader, saving_path=args.latent_saving_path)
    recons(model, test_loader)
