
import torch
import numpy as np

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized, min_val, max_val

def denormalize_data(normalized_data, min_val, max_val):
    return [x * (max_val - min_val) + min_val for x in normalized_data]

def generate_sequence_data(device):
    sequence = [
        # Pattern 1: Counting up by 1
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        
        # Pattern 2: Counting by 2
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        
        # Pattern 3: Counting by 3
        3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
        
        # Pattern 4: Counting by 5
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
        
        # Pattern 5: Alternating +10/-5
        10, 5, 15, 10, 20, 15, 25, 20, 30, 25,
        
        # Pattern 6: Symmetric around center
        2, 4, 6, 8, 10, 10, 8, 6, 4, 2,
        
        # Pattern 7: Powers of 2
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
        
        # Pattern 8: Fibonacci-like
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55,
        
        # Pattern 9: Squared numbers with sign alternating
        1, -4, 9, -16, 25, -36, 49, -64, 81, -100,
        
        # Pattern 10: Palindromic pattern
        5, 10, 15, 20, 25, 25, 20, 15, 10, 5
    ]
    
    normalized_sequence, min_val, max_val = normalize_data(sequence)
    
    X, y = [], []
    for i in range(len(normalized_sequence)-5):
        X.append(normalized_sequence[i:i+5])
        y.append(normalized_sequence[i+5])
    
    return torch.FloatTensor(X).unsqueeze(-1).to(device), torch.FloatTensor(y).to(device), sequence, min_val, max_val