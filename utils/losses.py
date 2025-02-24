import torch.nn as nn

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
    
    def forward(self, frame_t, frame_t_plus_1):
        # A simple implementation using L1 loss between consecutive frames
        return nn.functional.l1_loss(frame_t, frame_t_plus_1)

# You can add more custom loss functions here as needed.
