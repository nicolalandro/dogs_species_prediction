import pickle
import tarfile

from torch.serialization import _load, _open_zipfile_reader


def torch_load_targz(filep_ath):
    tar = tarfile.open(filep_ath, "r:gz")
    member = tar.getmembers()[0]
    with tar.extractfile(member) as untar:
        with _open_zipfile_reader(untar) as zipfile:
            torch_loaded = _load(zipfile, None, pickle)
    return torch_loaded


if __name__ == '__main__':
    torch_load_targz("../models/nts_net_state.tar.gz")
