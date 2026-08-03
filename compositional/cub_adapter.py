import numpy as np, torch, open_clip
from conic_lib import unit, cone_fit, cone_residual_deg, spherical_kmeans, vmf_distance_deg, auroc

@torch.no_grad()
def embed_clip(images, device="cuda"):
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(device).eval()
    feats = [model.encode_image(pre(im).unsqueeze(0).to(device)).cpu().numpy() for im in images]
    return unit(np.vstack(feats))

def split_by_attribute_pattern(embeds, attr_bits, hold_out_patterns):
    """attr_bits: (N, 312) binary. Group by a chosen attribute signature (e.g. a few
    salient attributes), hold out entire signatures as the 'unseen combination' set."""
    sig = tuple  # define your signature over a subset of attributes
    keys = [sig(row) for row in attr_bits]
    seen_mask   = np.array([k not in hold_out_patterns for k in keys])
    return embeds[seen_mask], embeds[~seen_mask]

# X_seen, X_unseen = split_by_attribute_pattern(embeds, attr_bits, held_out)
# X_ood = embeds of a disjoint species group (or a non-bird OOD set)
# then: identical cone_fit / spherical_kmeans / residual / auroc calls as the synthetic run
