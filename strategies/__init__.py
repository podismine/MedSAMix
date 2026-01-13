from .lfs import ViTLfsMerge
from .lfs_multiobj import ViTLfsMultiMerge

strategy_classes = {
    'lfs': ViTLfsMerge,
    'lfs_multiobj': ViTLfsMultiMerge
}
