import h5py
import numpy as np

path = '/home/ss-oss1/data/dataset/egocentric/ml-egodex/download/part1/color/0.hdf5'
with h5py.File(path, 'r') as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {name}/ (Group, keys: {list(obj.keys())})")
    print("Structure:")
    f.visititems(show)

    # 若 confidences 是 Group，看它下面有哪些 Dataset
    if 'confidences' in f:
        c = f['confidences']
        if isinstance(c, h5py.Group):
            print("\nconfidences is a Group. Sub-items:")
            for k in c.keys():
                sub = c[k]
                if isinstance(sub, h5py.Dataset):
                    arr = np.asarray(sub)
                    if np.issubdtype(arr.dtype, np.number):
                        print(f"  {k}: shape={arr.shape}, mean={float(np.mean(arr))}")
                    else:
                        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            arr = np.asarray(c)
            if np.issubdtype(arr.dtype, np.number):
                print("\nconfidences mean:", float(np.mean(arr)))