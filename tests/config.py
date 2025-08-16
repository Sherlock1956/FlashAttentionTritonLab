config = {
    "device": 'cuda',
    'vocab_size': 10000,
    'context_length': 1024,
    'd_model': 512,
    'num_layers': 4,
    'num_heads': 16,
    'd_ff': 1344,
    'rope_theta': 10000,
    'max_lr': 1e-3,
    'min_lr': 1e-6,
    'total_iterations': 160000,
    'warmup_iters': 4000,
    'batch_size': 8,
    'max_grad_norm': 5,
    'save_every': 10000,
    'val_every': 1000
}