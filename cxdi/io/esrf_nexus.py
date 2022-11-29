import hdf5plugin # needed for Eiger2 images
import h5py
from silx.io.h5py_utils import File as silx_File
import numpy as np

def is_h5group(obj):
    return isinstance(obj,h5py.Group)

def is_h5dataset(obj):
    return isinstance(obj,h5py.Dataset)

def group_str(h5group):
    keys = list(h5group.keys())
    keys.sort()
    return ",".join(keys)

def group_repr(h5group):
    keys = list(h5group.keys())
    keys.sort(key=lambda v: v.upper())
    if len(keys) == 0:
        return "Empty Group"
    nchars = max(map(len, keys))
    fmt = "%%%ds %%s" % (nchars)
    s = [
        "Group containing (sorted): ",
    ]
    for k in keys:
        #if len(k) == 0 or k[0] == "_":
        #    continue
        obj = getattr(h5group,k)
        if isinstance(obj, H5Dataset):
            value_str = str(obj)
        elif isinstance(obj, np.ndarray):
            value_str = "array, size %s, type %s" % (
                "x".join(map(str, obj.shape)),
                obj.dtype,
            )
        elif isinstance(obj, H5Group):
            value_str = str(obj)[:50]
        elif isinstance(obj, str):
            value_str = obj[:50].replace("\r\n", "\n").replace("\n", " ")
        elif isinstance(obj, (float, int)):
            value_str = "%g" % obj
        elif h5group[k] is None:
            value_str = "None"
        else:
            value_str = str(h5group[k]).replace("\r\n", "\n").replace("\n", " ")
        if len(str(obj)) > 50:
            value_str += " ..."
        s.append(fmt % (k, value_str))
    return "\n".join(s)

def find_subscans(esrf_nexus,scan):
    toret=[]
    for subscan in range(1,10):
        if f"{scan}.{subscan}" in esrf_nexus.scans:
            toret.append(subscan)
    return toret


class H5Dataset:
    def __init__(self,dataset):
        self._h5handle = dataset
        # "export" some method
        methods = "shape","dtype","size"
        for method in methods:
            h5py_dataset_method = getattr(self._h5handle,method)
            setattr(self,method,h5py_dataset_method)

        if self.size < 10_000:
            self._data = dataset[()]
        else:
            self._data = None

    def __getitem__(self,key):
        if self._data is None:
            if key == ():
                self._data = self._h5handle[()] # caching read data
                return self._data
            else:
                return self._h5handle[key]
        else:
            return self._data[key]

    def __str__(self):
        if isinstance(self._data,str):
            s = self._data
        else:
            s = f"shape={self.shape}, type={self.dtype}"
            if self._data is not None:
                s += " " + str(self._data)
        return s[:50]

    def __repr__(self):
        if self.size < 10:
            s = "array: " + ",".join(map(str,[v for v in self]))
            return s
        else:
            return self.__str__()


class H5Group:
    def __init__(self,h5group):
        self._h5handle = h5group
        if not is_h5group(h5group):
            return h5group
        keys = list(h5group.keys())
        self._keys = keys

#        methods = "items","keys","values"
#        for method in methods:
#            h5py_group_method = getattr(self._h5handle,method)
#            setattr(self,method,h5py_group_method)

        for key,value in h5group.items():
            if is_h5group(value):
                # check if dataset in /value
                if "value" in value and len(value) == 1:
                    value = H5Dataset(value["value"])
                else:
                    value = H5Group(value)
            if is_h5dataset(value):
                value = H5Dataset(value)
            setattr(self,key,value)

    def __str__(self):
        return group_str(self)

    def __repr__(self):
        return group_repr(self)

    def keys(self):
        return self._keys

    def __getitem__(self,key):
        return getattr(self,key)
        #if hasattr(self,key):
        #    return getattr(self,key)
        #else:
        #    return self._h5handle[key]

class ESRFSubScan:
    def __init__(self,esrf_nexus,scan_num,subscan):
        self._scan_num = scan_num
        self._subscan = subscan
        self._data = H5Group(esrf_nexus.h5_handle[f"{scan_num}.{subscan}"])

        for key in self._data._keys:
            setattr(self,key,self._data[key])

    def __getitem__(self,k):
        return self._data[k]

    def __str__(self):
        s = f"scan {self._scan_num}.{self._subscan} {str(self._data.title)}"
        return s

    def __repr__(self):
        return self.__str__()


class ESRFScan:
    def __init__(self,esrf_nexus,scan_num):
        self._scan_num = scan_num
        self._subscans = find_subscans(esrf_nexus,scan_num)
        self._data = [ESRFSubScan(esrf_nexus,scan_num,s) for s in self._subscans]

    def __str__(self):
        s_list = [str(d) for d in self._data]
        return "\n".join(s_list)

    def __repr__(self):
        return self.__str__()



class ESRFNexus:
    def __init__(self,fname):
        self.fname = fname
        # use str(fname) below because it might be a read_files.Dataclass
        self.h5_handle = silx_File(str(fname),"r")
        # scans include subscans 1.1, 1.2, 1.3, 2.1, ...
        self.scans = list(self.h5_handle.keys())
        scans_no = [int(s.split(".")[0]) for s in self.scans]
        scans_no = list(set(scans_no))
        self.scans_no = scans_no
        self._data = dict()

    def __getitem__(self,scan):
        if isinstance(scan,int):
            subscan = 1
        else:
            scan,subscan = map(int,scan.split("."))
        scan_str = f"{scan}.{subscan}"
        if scan_str in self.scans:
            if not scan_str in self._data: self._data[scan_str] = ESRFSubScan(self,scan,subscan)
            return self._data[scan_str]
        else:
            print(f"ESRFNexus, scan {scan_str} does not exist in file")

    def list_scans(self):
        for scan in self.scans_no:
            scan_str = f"{scan}.1"
            print(f"{scan}.1 {str(self[scan_str]._data.title)}")

    def __str__(self):
        return f"ESRF Nexus data, scans {self.scans}"

    def __repr__(self):
        return self.__str__()


