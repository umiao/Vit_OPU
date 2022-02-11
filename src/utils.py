"""utils.py - Helper functions
"""

import numpy as np
import torch
from torch.utils import model_zoo

from .configs import PRETRAINED_MODELS


def load_pretrained_weights(
    model, 
    model_name=None, 
    weights_path=None, 
    load_first_conv=True, 
    load_fc=True, 
    load_repr_layer=False,
    resize_positional_embedding=False,
    verbose=True,
    strict=True,
):
    """Loads pretrained weights from weights path or download using url.
    Args:
        model (Module): Full model (a nn.Module)
        model_name (str): Model name (e.g. B_16)
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_first_conv (bool): Whether to load patch embedding.
        load_fc (bool): Whether to load pretrained weights for fc layer at the end of the model.
        resize_positional_embedding=False,
        verbose (bool): Whether to print on completion
    """
    assert bool(model_name) ^ bool(weights_path), 'Expected exactly one of model_name or weights_path'
    
    # Load or download weights
    if weights_path is None:
        url = PRETRAINED_MODELS[model_name]['url']
        if url:
            state_dict = model_zoo.load_url(url)
        else:
            raise ValueError(f'Pretrained model for {model_name} has not yet been released')
    else:
        state_dict = torch.load(weights_path)

    # Modifications to load partial state dict
    expected_missing_keys = []
    if not load_first_conv and 'patch_embedding.weight' in state_dict:
        expected_missing_keys += ['patch_embedding.weight', 'patch_embedding.bias']
    if not load_fc and 'fc.weight' in state_dict:
        expected_missing_keys += ['fc.weight', 'fc.bias']
    if not load_repr_layer and 'pre_logits.weight' in state_dict:
        expected_missing_keys += ['pre_logits.weight', 'pre_logits.bias']
    for key in expected_missing_keys:
        state_dict.pop(key)

    # Change size of positional embeddings
    if resize_positional_embedding: 
        posemb = state_dict['positional_embedding.pos_embedding']
        posemb_new = model.state_dict()['positional_embedding.pos_embedding']
        state_dict['positional_embedding.pos_embedding'] = \
            resize_positional_embedding_(posemb=posemb, posemb_new=posemb_new, 
                has_class_token=hasattr(model, 'class_token'))
        maybe_print('Resized positional embeddings from {} to {}'.format(
                    posemb.shape, posemb_new.shape), verbose)

    # Load state dict
    ret = model.load_state_dict(state_dict, strict=False)
    if strict:
        assert set(ret.missing_keys) == set(expected_missing_keys), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        assert not ret.unexpected_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)
        maybe_print('Loaded pretrained weights.', verbose)
    else:
        maybe_print('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), verbose)
        maybe_print('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), verbose)
        return ret


def maybe_print(s: str, flag: bool):
    if flag:
        print(s)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def resize_positional_embedding_(posemb, posemb_new, has_class_token=True):
    """Rescale the grid of position embeddings in a sensible manner"""
    from scipy.ndimage import zoom

    # Deal with class token
    ntok_new = posemb_new.shape[1]
    if has_class_token:  # this means classifier == 'token'
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

    # Get old and new grid sizes
    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    # Rescale grid
    zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = zoom(posemb_grid, zoom_factor, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    posemb_grid = torch.from_numpy(posemb_grid)

    # Deal with class token and return
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

import math
def auto_quantize(input_v):
    # if type(input_v) != type(np.array([])):
    input_v = input_v.cpu().numpy()
    max_v = input_v.max()
    min_v = input_v.min()
    max_v = math.ceil(max_v)
    min_v = math.floor(min_v)
    int_part = max(abs(max_v), abs(min_v))
    int_bits = math.ceil(np.log2(int_part)) if int_part != 0 else 0
    mantisaa_bits = 16 - 1 - int_bits
    return mantisaa_bits

def float2fix_np(value, frac_len=0, word_len=8, round_method='floor'):
    min_value = -2 ** (word_len - 1)
    max_value = 2 ** (word_len - 1) - 1

    if round_method == 'round':
        fix_value = np.floor(value * (2 ** frac_len) + 0.5)
    else:
        fix_value = np.floor(value * (2 ** frac_len))

    fix_value[fix_value < min_value] = min_value
    fix_value[fix_value > max_value] = max_value
    fix_value = fix_value / (2 ** frac_len)
    return fix_value


def float2fix(value, frac_len):
    value = torch.mul(value, 1<<frac_len)
    value = torch.clamp(value, min=-2 ** 15, max=2 ** 15 - 1)
    value = torch.floor(value)
    value = torch.div(value, 1<<frac_len)
    return value

global_quantize_param_dict = dict()

def sqrt_cordic ( x, n = 12 ):
  from sys import exit

  if ( x < 0.0 ):
    print ( '' )
    print ( 'SQRT_CORDIC - Fatal error!' )
    print ( '  X < 0.' )
    exit ( 'SQRT_CORDIC - Fatal error!' )

  if ( x == 0.0 ):
    y = 0.0
    return y

  if ( x == 1.0 ):
    y = 1.0
    return y

  poweroftwo = 1.0

  if ( x < 1.0 ):

    while ( x <= poweroftwo * poweroftwo ):
      poweroftwo = poweroftwo / 2.0

    y = poweroftwo

  elif ( 1.0 < x ):

    while ( poweroftwo * poweroftwo <= x ):
      poweroftwo = 2.0 * poweroftwo

    y = poweroftwo / 2.0

  for i in range ( 0, n ):
    poweroftwo = poweroftwo / 2.0
    if ( ( y + poweroftwo ) * ( y + poweroftwo ) <= x ):
      y = y + poweroftwo

  return y


def arctan_cordic(x, y, n=64):
    import numpy as np

    angles = np.array([ \
        7.8539816339744830962E-01, \
        4.6364760900080611621E-01, \
        2.4497866312686415417E-01, \
        1.2435499454676143503E-01, \
        6.2418809995957348474E-02, \
        3.1239833430268276254E-02, \
        1.5623728620476830803E-02, \
        7.8123410601011112965E-03, \
        3.9062301319669718276E-03, \
        1.9531225164788186851E-03, \
        9.7656218955931943040E-04, \
        4.8828121119489827547E-04, \
        2.4414062014936176402E-04, \
        1.2207031189367020424E-04, \
        6.1035156174208775022E-05, \
        3.0517578115526096862E-05, \
        1.5258789061315762107E-05, \
        7.6293945311019702634E-06, \
        3.8146972656064962829E-06, \
        1.9073486328101870354E-06, \
        9.5367431640596087942E-07, \
        4.7683715820308885993E-07, \
        2.3841857910155798249E-07, \
        1.1920928955078068531E-07, \
        5.9604644775390554414E-08, \
        2.9802322387695303677E-08, \
        1.4901161193847655147E-08, \
        7.4505805969238279871E-09, \
        3.7252902984619140453E-09, \
        1.8626451492309570291E-09, \
        9.3132257461547851536E-10, \
        4.6566128730773925778E-10, \
        2.3283064365386962890E-10, \
        1.1641532182693481445E-10, \
        5.8207660913467407226E-11, \
        2.9103830456733703613E-11, \
        1.4551915228366851807E-11, \
        7.2759576141834259033E-12, \
        3.6379788070917129517E-12, \
        1.8189894035458564758E-12, \
        9.0949470177292823792E-13, \
        4.5474735088646411896E-13, \
        2.2737367544323205948E-13, \
        1.1368683772161602974E-13, \
        5.6843418860808014870E-14, \
        2.8421709430404007435E-14, \
        1.4210854715202003717E-14, \
        7.1054273576010018587E-15, \
        3.5527136788005009294E-15, \
        1.7763568394002504647E-15, \
        8.8817841970012523234E-16, \
        4.4408920985006261617E-16, \
        2.2204460492503130808E-16, \
        1.1102230246251565404E-16, \
        5.5511151231257827021E-17, \
        2.7755575615628913511E-17, \
        1.3877787807814456755E-17, \
        6.9388939039072283776E-18, \
        3.4694469519536141888E-18, \
        1.7347234759768070944E-18])

    x1 = x
    y1 = y
    #
    #  Account for signs.
    #
    if (x1 < 0.0 and y1 < 0.0):
        x1 = - x1
        y1 = - y1

    if (x1 < 0.0):
        x1 = - x1
        sign_factor = -1.0
    elif (y1 < 0.0):
        y1 = -y1
        sign_factor = -1.0
    else:
        sign_factor = +1.0

    theta = 0.0
    poweroftwo = 1.0

    for j in range(0, n):

        if (y1 <= 0.0):
            sigma = +1.0
        else:
            sigma = -1.0

        if (j < angles.size):
            angle = angles[j]
        else:
            angle = angle / 2.0

        x2 = x1 - sigma * poweroftwo * y1
        y2 = sigma * poweroftwo * x1 + y1

        theta = theta - sigma * angle

        x1 = x2
        y1 = y2

        poweroftwo = poweroftwo / 2.0

    theta = sign_factor * theta

    return theta