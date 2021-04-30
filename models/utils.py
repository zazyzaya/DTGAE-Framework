import torch.distributed.rpc as rpc 
import torch.multiprocessing as mp

# Provided by torch in possibly next update for the RPC API 
# but for now, we need to add these ourselves
def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=60*15)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=60*15)

'''
Because there are some remote parameters in the model,
just calling params() will confuse the optimiser. Instead
we create an RRef for each parameter to tell the opt where
to find it
'''
def _param_rrefs(module):
    rrefs = []
    for param in module.parameters():
        rrefs.append(
            rpc.RRef(param)
        )
    
    return rrefs