# cxdi
Python based tools for coherent scattering, imaging and dynamics

## reading (ESRF) files ##

```python
from cxdi import io
data = io.ESRFNexus(fname)
# data = io.get_dataset(experiment="ch6495",sample_name="alignment",datasetnumber=1)

# list scans
data.list_scans()

# retrieve scan
scan = data[42]
```
