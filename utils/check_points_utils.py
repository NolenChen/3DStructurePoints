import torch
import os


def save_checkpoint(filename, model_3d=None, model_2d=None, optimizer=None, iters=None, epoch=None, meta_data=None):
    print("save checkpoint '{}'".format(filename))
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model_3d is not None:
        if isinstance(model_3d, torch.nn.DataParallel):
            model_state_3d = model_3d.module.state_dict()
        else:
            model_state_3d = model_3d.state_dict()
    else:
        model_state_3d = None

    if model_2d is not None:
        if isinstance(model_2d, torch.nn.DataParallel):
            model_state_2d = model_2d.module.state_dict()
        else:
            model_state_2d = model_2d.state_dict()
    else:
        model_state_2d = None

    state = {
        'iter': iters,
        'epoch': epoch,
        'model_state_2d': model_state_2d,
        'model_state_3d': model_state_3d,
        'optimizer_state': optim_state,
        'meta_data': meta_data
    }
    torch.save(state, filename)


def load_checkpoint(filename, model_3d=None, optimizer=None, meta_data=None):

    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        iters = checkpoint.get('iter', 0.0)
        epoch = checkpoint['epoch']
        if model_3d is not None and checkpoint['model_state_3d'] is not None:
            model_3d.load_state_dict(checkpoint['model_state_3d'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if meta_data is not None and 'meta_data' in checkpoint:
            for key in checkpoint['meta_data']:
                meta_data[key] = checkpoint['meta_data'][key]
        return epoch, iters
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None

