import torch

# distance functions
def euclid_dist(x, y):
    """
    Euclidean distance between two tensors.
    euclid_dist(x) = sqrt(x^2 + y^2)
    """
    return torch.dist(x, y, p=2)

def arcosh(x):
    """
    arcosh(x) = ln(x + sqrt(x^2 - 1))
    elementwise arcosh operation.
    """
    return torch.log(x + torch.sqrt(torch.add(torch.pow(x, 2), -1.)))

def hyp_dist(u, v):
    """
    Hyperbolic distance between two tensors.
    hyp_dist(x) = arcosh(1 + 2*|u - v|^2 / ((1 - |u|^2) * (1 - |v|^2)))    
    """
    # u_norm = 1 - |u|^2
    u_norm = torch.add((torch.neg(torch.pow(torch.norm(u, 2, 1, keepdim=True), 2))), 1.)
    # v_norm = 1 - |v|^2
    v_norm = torch.add((torch.neg(torch.pow(torch.norm(v, 2, 1, keepdim=True), 2))), 1.)
    # delta is the isometric invariants, del = 2*|u - v|^2 / (u_norm * v_norm)                            
    delta = torch.mul(torch.pow(torch.dist(u, v), 2), 2.) / torch.mul(u_norm, v_norm)

    hyper_dist = arcosh(1 + delta)
    assert hyper_dist.size() == (1, 1)
    return hyper_dist

def proj(x, eps=0.000001):
    #norm = torch.norm(x, 2, 0)
    #ones = torch.ones_like(x)
    
    # 1 if ||x|| >= 1, 0 otherwise
    #cmp = torch.ge(norm, ge)
    
    #normalised = torch.div(x, norm)
    
    # normalize x in rows dim
    return torch.renorm(x, 2, 0, 1.)

def inverse_metric_tensor(theta):
    """
    gives the inverse metric tensor at theta. 
    Use this to multiply euclidean gradients and convert to riemannian gradients.
    Input:
        theta - the embedding vectors, shape = (batch_size, ndim).
    Return:
        the scalar = (1 - ||theta||^2)^2 / 4, shape = (batch_size, 1).
    """
    # || theta ||
    norm = torch.norm(theta, 2, 1, keepdim=True)
    one = torch.ones(norm.size())
    norm = torch.pow(norm, 2)

    # 1 - ||theta||^2
    ret = torch.ones(norm.size()) - torch.pow(norm.data, 2) # note the data here, use tensor instead of variable

    # (1 - ||theta||^2)^2 / 4
    return torch.div(torch.pow(ret, 2), 4.)