"""Token-count -> feature-map (H, W) lookup used by the attention masks."""
import math


# get H,W at unet depth
def build_S_to_HW(H_full, W_full, min_size=4):
    """
    Build a dictionary mapping total token count (2 * n_tokens) -> (H_feat, W_feat),
    given the full image size (H_full, W_full). Allows lookup via S_to_HW[n*2].

    Args:
        H_full (int): full image height
        W_full (int): full image width
        min_size (int): smallest spatial resolution (H_feat or W_feat) to include

    Returns:
        dict: S_to_HW[n_tokens*2] = (H_feat, W_feat)
    """
    S_to_HW = {}
    H, W = H_full, W_full
    while H >= min_size and W >= min_size:
        n_tokens = H * W
        S_to_HW[n_tokens * 2] = (H, W)  # *2 because we have 2 views
        H = math.ceil(H / 2)
        W = math.ceil(W / 2)

    H, W = H_full, W_full
    while H >= min_size and W >= min_size:
        n_tokens = H * W
        S_to_HW[n_tokens * 2] = (H, W)  # *2 because we have 2 views
        H = H // 2
        W = W // 2

    # adds both ceil and floor, since both occur.
    S_to_HW['full_image'] = (H_full, W_full)
    return S_to_HW
