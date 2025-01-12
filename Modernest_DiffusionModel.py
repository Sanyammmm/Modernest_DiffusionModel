import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),  # +1 for time embedding
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t):
        # Expand time to match image dimensions
        t = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t], dim=1)  # Add time channel
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decoder
        d1 = self.dec1(e2)
        output = self.dec2(d1)
        
        return output

class DiffusionModel:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleUNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def add_noise(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        ε = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * ε, ε

    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=self.device)
        
        # Add noise to the input
        noisy_x, noise = self.add_noise(x, t)
        
        # Predict noise
        predicted_noise = self.model(noisy_x, t / self.timesteps)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples, size=(3, 32, 32)):
        self.model.eval()
        x = torch.randn(n_samples, *size).to(self.device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.ones(n_samples, device=self.device).long() * t
            predicted_noise = self.model(x, t_batch / self.timesteps)
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
            
        return x.clamp(-1, 1)

# Example usage
def main():
    # Setup data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                         download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    diffusion = DiffusionModel()
    
    # Training loop
    n_epochs = 5
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            images = batch[0].to(diffusion.device)
            loss = diffusion.train_step(images)
            total_loss += loss
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 1 == 0:
            samples = diffusion.sample(4)
            # Changed 'range' to 'value_range' to fix the error
            torchvision.utils.save_image(samples, f"samples_epoch_{epoch+1}.png",
                                       normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    main()
