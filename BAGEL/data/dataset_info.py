# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset, ReconstructionDataset
from .referring import ReferringDataset
from .chartqapro import ChartQAProDataset
from .vlm_dataset import SftJSONLIterableDataset
from .segment import SegmentDataset
from .panoptic import PanopticDataset
from .denoise import DenoiseDataset
from .edge import EdgeDataset
from .semantic import SemanticDataset
from .super_resolution import SuperResolutionDataset
from .inpainting import InpaintingDataset
from .deblurring import DeblurringDataset
from .enhance import EnhanceDataset
from .derain import DerainDataset
from .instance import InstanceDataset
from .detect import DetectDataset
from .reca import ReconstructDataset
from .view import ViewDataset
from .panoptic_100k import Panoptic100KDataset
from .sam_190k import Sam190KDataset
from .step_sft import STEPSftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'reconstruction': ReconstructionDataset,
    'referring': ReferringDataset,
    'chartqapro': ChartQAProDataset,
    'segment': SegmentDataset,
    'panoptic2k': PanopticDataset,
    'denoise2k': DenoiseDataset,
    'edge2k': EdgeDataset,
    'semantic2k': SemanticDataset,
    # Added: super_resolution | inpainting | deblurring | enhance2k | derain2k
    'super_resolution2k': SuperResolutionDataset,
    'inpainting2k': InpaintingDataset,
    'deblurring2k': DeblurringDataset,
    'enhance2k': EnhanceDataset,
    'derain2k': DerainDataset,
    # Added: instance
    'instance2k': InstanceDataset,
    'detect2k': DetectDataset,
    'reconstruct2k': ReconstructDataset,
    'view': ViewDataset,
    'panoptic100k': Panoptic100KDataset,
    'sam190k': Sam190KDataset,
    'step_sft': STEPSftJSONLIterableDataset,
}


# NOTE:
# All paths below are placeholders relative to the project root (or an arbitrary
# data root). Please replace them with the actual locations of your datasets
# before running training / evaluation. We use POSIX-style relative paths and
# avoid any absolute paths that are specific to our internal environment.
DATASET_INFO = {
    # 't2i_pretrain': {
    #     't2i': {
    #         'data_dir': 'data/bagel_example/t2i',                 # path of the parquet files
    #         'num_files': 10,                                      # number of data units to be sharded across all ranks and workers
    #         'num_total_samples': 1000,                            # number of total samples in the dataset
    #     },
    # },
    # 'unified_edit': {
    #     'seedxedit_multi': {
    #         'data_dir': 'data/bagel_example/editing/seedxedit_multi',
    #         'num_files': 10,
    #         'num_total_samples': 1000,
    #         'parquet_info_path': 'data/bagel_example/editing/parquet_info/seedxedit_multi_nas.json',
    #     },
    # },
    'vlm_sft': {
        'vlm_sft': {
            'data_dir': 'data/LLaVA-OneVision-SGT/llava_onevision_balanced_500k/images',
            'jsonl_path': 'data/LLaVA-OneVision-SGT/llava_onevision_balanced_500k/annotations/all_data.jsonl',
            'num_total_samples': 500000,
        },
    },
    'step_sft': {
        'step_sft': {
            'data_dir': '',
            'jsonl_path': 'data/step3_5_sft/json/general/step_sft_90k.jsonl',
            'num_total_samples': 90000,
        },
    },
    'reconstruction': {
        'webdataset': {
            'data_dir': 'data/reconstruction/train',    # directory containing all tar files
            'num_files': 1,                             # number of data units to be sharded across all ranks and workers
            'cache_dir': 'data/reconstruction/train',   # cache directory for extracted images
        },
    },
    'chartqapro': {
        'chartqapro': {
            'data_dir': 'data/ChartQAPro',
            'num_files': 1,
            'jsonl_path': 'data/ChartQAPro/chartqapro.jsonl',
            'cache_dir': 'data/ChartQAPro',
        },
    },
    'view': {
        'view': {
            'data_dir': 'data/sam-qa/',
            'num_files': 1,
            'jsonl_path': 'data/marigold/vkitti/vkitti.txt',
            'cache_dir': 'data/sam-qa/',
        },
    },
    'panoptic2k': {
        'panoptic2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'panoptic100k': {
        'panoptic100k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco100K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'sam190k': {
        'sam190k': {
            'data_dir': 'data/sam-qa/file_names/sam_selection/',
            'num_files': 1,
            'jsonl_path': 'data/sam-qa/file_names/sam_190k_new.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'edge2k': {
        'edge2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'semantic2k': {
        'semantic2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'detect2k': {
        'detect2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'reconstruct2k': {
        'reconstruct2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'instance2k': {
        'instance2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco_instance2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'denoise2k': {
        'denoise2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'derain2k': {
        'derain2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'inpainting2k': {
        'inpainting2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'enhance2k': {
        'enhance2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'deblurring2k': {
        'deblurring2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'super_resolution2k': {
        'super_resolution2k': {
            'data_dir': 'data/coco/train2017/',
            'num_files': 1,
            'jsonl_path': 'data/coco/coco2K.txt',
            'cache_dir': 'data/coco/train2017/',
        },
    },
    'referring': {
        'omniedit': {
            'data_dir': 'data/omniedit/images/',
            'num_files': 1,
            'jsonl_path': 'data/omniedit-reca/multiple_repaint_info_pro.jsonl',
            # 'cache_dir': 'data/omniedit/train',   # cache directory for extracted images
        },
    },
}
