import os
import glob
import pydicom
from typing import List, Tuple

def list_dicom_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, "*.dcm")
    return sorted(glob.glob(pattern))

def filter_by_patient_id(paths: List[str], query: str) -> List[str]:
    results = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            pid = getattr(ds, 'PatientID', '')
            if query.lower() in pid.lower():
                results.append(p)
        except Exception:
            continue
    return results


def read_dicom_raw(path: str) -> Tuple[bytes, str]:
    with open(path, 'rb') as f:
        raw = f.read()
    name = os.path.basename(path)
    return raw, name


def read_dicom_metadata(path: str) -> dict:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return {
            'PatientID': getattr(ds, 'PatientID', None),
            'StudyDate': getattr(ds, 'StudyDate', None),
            'Modality': getattr(ds, 'Modality', None),
            'StudyDescription': getattr(ds, 'StudyDescription', None),
        }
    except Exception as e:
        return {}
