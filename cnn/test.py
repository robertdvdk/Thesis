"""
Module-level docstring
"""
# Import statements
import torch
# Function definitions

if __name__ == "__main__":
    """The main function of this module"""
    testin = 'ACGTTC'

    print(testin)
    newtest = torch.Tensor([[2., 3., 4.], [5., 6., 7.]])
    print(newtest)
    torch.cat(tensors=(testin, newtest.reshape(1, 2, 3)), dim=0)
    print(testin)

