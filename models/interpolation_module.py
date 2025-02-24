import torch
import torch.nn as nn
import torch.nn.functional as F

class InterpolationModule(nn.Module):
    def __init__(self, in_channels=3, base_filters=32):
        super(InterpolationModule, self).__init__()
        # A shallow network to estimate motion/displacement map between keyframes
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 2, kernel_size=3, padding=1)  # 2 channels: displacement in x and y directions
        )
        # Refinement CNN to blend the warped keyframes
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, keyframe1, keyframe2, t=0.5):
        # Concatenate keyframes along the channel dimension to predict motion
        combined = torch.cat([keyframe1, keyframe2], dim=1)
        flow = self.motion_estimator(combined)
        
        # Create a normalized grid for warping (batch size, height, width, 2)
        B, C, H, W = keyframe1.size()
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W))
        grid = torch.stack((grid_x, grid_y), 2).to(keyframe1.device)  # shape: [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Adjust grid with flow scaled by t (interpolation factor)
        flow_scaled = flow.permute(0, 2, 3, 1) * t
        grid_warp = grid + flow_scaled
        # Warp keyframe1 towards keyframe2
        warped_keyframe1 = F.grid_sample(keyframe1, grid_warp, align_corners=True)
        
        # Blend the warped keyframe and keyframe2
        blended = torch.cat([warped_keyframe1, keyframe2], dim=1)
        refined_frame = self.refinement(blended)
        return refined_frame

if __name__ == '__main__':
    # Quick test of the interpolation module
    module = InterpolationModule()
    dummy_keyframe1 = torch.randn(1, 3, 64, 64)
    dummy_keyframe2 = torch.randn(1, 3, 64, 64)
    interpolated_frame = module(dummy_keyframe1, dummy_keyframe2, t=0.5)
    print("Interpolated frame shape:", interpolated_frame.shape)
