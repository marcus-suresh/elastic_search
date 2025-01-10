# imports are always needed
import torch

# get index of currently selected device
torch.cuda.current_device() # returns 0 in my case


# get number of GPUs available
torch.cuda.device_count() # returns 1 in my case


# get the name of the device
torch.cuda.get_device_name(0) # good old Tesla K80

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

