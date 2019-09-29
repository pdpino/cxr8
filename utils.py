from datetime import datetime
import time

ALL_DISEASES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
]

DISEASE_INDEX = { name: index for index, name in enumerate(ALL_DISEASES) }

def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
