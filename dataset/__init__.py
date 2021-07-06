from .dataset import SPGISpeechDataset, collate_fun, PreProcess
from .iwslt_dataset import IWSLTDataset

BuildDataset = {
                'spgispeech': SPGISpeechDataset,
                'iwslt': IWSLTDataset
}