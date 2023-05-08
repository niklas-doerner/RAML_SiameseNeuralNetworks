import torch
import torch.nn as nn


# modify the edge detector kernel in such a way that
# it calculates the derivatives in x and y direction
edge_detector_kernel = torch.zeros(2, 1, 2, 2) #𝐶out×𝐶in×ℎ×𝑤 #out_channels=2 #in_channels=1 #kernel_size=(2, 2)

# input grayscale image 𝑢 of size 1×1×𝐻×𝑊
# derivative of the input image 
# in 𝑦 direction in its --> first output channel ==> edge_detector_kernel[0,0,:,:]
# in 𝑥 direction in its --> second output channel => edge_detector_kernel[1,0,:,:]

# first channel output 𝑣0,𝑖,𝑗 = 𝑢0,𝑖+1,𝑗 − 𝑢0,𝑖,𝑗 
# second channel output 𝑣1,𝑖,𝑗 = 𝑢0,𝑖,𝑗+1 − 𝑢0,𝑖,𝑗
# for 𝑖=0,…,𝐻−1 -> 0,1 
# for 𝑗=0,…,𝑊−1 -> 0,1

edge_detector_kernel[0,0,:,:] = torch.tensor([[-1,0],[1,0]])
edge_detector_kernel[1,0,:,:] = torch.tensor([[-1,1],[0,0]])

class Conv2d(nn.Module):
    
    def __init__(self, kernel, padding=0, stride=1):
        super().__init__()
        self.kernel = nn.Parameter(kernel)
        self.padding = ZeroPad2d(padding)
        self.stride = stride
        
    def forward(self, x):
        x = self.padding(x)
        # For input of shape C x H x W
        C, H, W = x.shape 
        
        # weights for a convolution layer 𝐶𝑜𝑢𝑡×𝐶𝑖𝑛×ℎ×𝑤
        # 𝐶in×𝐻×𝑊 -> w.r.t. a kernel of size 𝐶out×𝐶in×ℎ×𝑤 => 𝐶out×𝐻′×𝑊′
        # implement the convolution of x with self.kernel
        Cin, Cout, h, w = self.kernel.shape
        
        #𝐻′=𝐻−(ℎ−1) and 𝑊′=𝑊−(𝑤−1)
        H_ = H - h - 1
        W_ = W - w - 1
        
        # The output is expected to be of size C x H' x W'    
        # output values 𝑧 for input 𝑥 
        z = torch.zeros(Cout,H_,W_)
        
        # 𝜏(𝑥∗𝑘)
        k = self.kernel
        for Hi in range(H_), Wi in range(W_):
            x_tau = x[:, Hi:Hi+h, Wi:Wi+w] * k
            z[:,Hi,Wi] = x_tau.sum
            
        return z
        
        # using self.stride as stride
             
        
class ZeroPad2d(nn.Module):
    
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
        
    def forward(self, x):
        # For input of shape C x H x W
        C, H, W = x.shape 

        #such that the output is of size
        # C x (H + 2 * self.padding) x (W + 2 * self.padding)
        H_padding = H + 2 * self.padding
        W_padding = W + 2 * self.padding
        X_padding = torch.zeros(C, H_padding, W_padding)
        
        # return tensor zero padded equally at left, right,
        # top, bottom 
        
        # between H / W and padding x value, else zeros => 0x0 = X with zero padding
        X_padding[:, self.padding:self.padding+H, self.padding:self.padding+W] = X_padding
        return X_padding
