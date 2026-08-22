import json
import numpy as np

def safe(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(v) for v in value]
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return str(value)

def run():
    obj={"array":np.arange(6).reshape(2,3),"scalar":np.float64(1.25)}
    data=json.loads(json.dumps(safe(obj)))
    assert data["array"]==[[0,1,2],[3,4,5]]
    assert data["scalar"]==1.25
    print("V11 JSON NUMPY SERIALIZATION TEST PASSED")

if __name__=="__main__":
    run()
