import torch 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE


def visualize_tsne_vae_output(model, test_loader, config , saving_path, num_samples=5000, perplexity=30, learning_rate=200):
    """
    Visualize clustering of VAE output using t-SNE.

    Parameters:
    - model: Trained VAE model.
    - test_loader: DataLoader for the test data.
    - config: Configuration dictionary (containing device info).
    - num_samples: Number of samples to use for t-SNE (default=1000).
    - perplexity: Perplexity parameter for t-SNE (default=30).
    - learning_rate: Learning rate parameter for t-SNE (default=200).

    Returns:
    - None (plots the t-SNE visualization).
    """
    # Collect the latent space vectors from the VAE model
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Get the latent representation from the VAE
            inputs, batch_labels = batch[0], batch[1]  # Assuming inputs and labels are present
            z = model(inputs.to(config['device']))  
            z = z.x_recon
            latent_vectors.append(z.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())

            # Stop collecting after num_samples is reached
            if len(latent_vectors) * z.shape[0] >= num_samples:
                break
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Apply t-SNE on the latent vectors
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate)
    tsne_result = tsne.fit_transform(latent_vectors[:num_samples])

    # Plot the result
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels[:num_samples], cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of VAE final output ")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(saving_path)
    plt.show()
    
    
def visualize_vae_latent_variable(model , train_loader  , device ,saving_path) : 
    # for latent variabel 
    # encode and plot the z values for the train set 
    model.eval()
    z_all = []
    y_all = []
    with torch.no_grad():
        for data, target in tqdm(train_loader, desc='Encoding'):
            data = data.to(device)
            output = model(data,loss=False)
            z_all.append(output.z_sample.cpu().numpy())
            y_all.append(target.numpy())
    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap='tab10')
    plt.colorbar()
    plt.savefig(saving_path)
    plt.show()
    
    
# some image visulization 
def recons(model, test_loader,config , saving_path ,  samples=9): 
    model.eval() 
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(9, 9))  # Set a figure size
    axis = ax.flatten()
    
    with torch.no_grad():  
        # Get a batch of images from the test loader
        batch = next(iter(test_loader))[0][:samples]  # Get the first batch and select the number of samples
        for idx, img in enumerate(batch): 
            img = img.to(config['device'])  # Move image to the appropriate device
            x_recon = model(img, loss=False).x_recon.reshape( 28, 28)  # Reshape the reconstructed output
            
            # Ensure that the reconstructed image is in the right format for imshow
            axis[idx].imshow(x_recon.cpu().numpy(), cmap='gray')  # Use imshow to visualize the image
            axis[idx].axis('off')  # Hide the axis for clarity
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(saving_path)
    plt.show()  # Show the plot
        