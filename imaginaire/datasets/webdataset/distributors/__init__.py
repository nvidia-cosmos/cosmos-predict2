from imaginaire.datasets.webdataset.distributors.basic import ShardlistBasic
from imaginaire.datasets.webdataset.distributors.multi_aspect_ratio import ShardlistMultiAspectRatio
from imaginaire.datasets.webdataset.distributors.multi_aspect_ratio_v2 import ShardlistMultiAspectRatioInfinite

distributors_list = {
    "basic": ShardlistBasic,
    "multi_aspect_ratio": ShardlistMultiAspectRatio,
    "multi_aspect_ratio_infinite": ShardlistMultiAspectRatioInfinite,
}
