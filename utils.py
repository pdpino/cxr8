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
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

def duration_to_str(all_seconds):
    seconds = all_seconds % 60
    minutes = all_seconds // 60
    hours = minutes // 60
    
    minutes = minutes % 60

    return "{}h {}m {}s".format(hours, minutes, seconds)