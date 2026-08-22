import math

class FakeRadia:
    def __init__(self, fail_relax=False):
        self.next_id=1; self.objects={}; self.materials={}; self.obj_material={}; self.fail_relax=fail_relax
        self.applied=[]; self.divided=[]; self.relaxed=False
    def _id(self):
        i=self.next_id; self.next_id+=1; return i
    def ObjRecMag(self,c,s,m):
        if any(float(v)<=0 for v in s): raise RuntimeError("bad size")
        i=self._id(); self.objects[i]={"type":"mag","c":list(c),"s":list(s),"m":list(m)}; return i
    def ObjCnt(self,objs):
        if not objs: raise RuntimeError("empty container")
        i=self._id(); self.objects[i]={"type":"cnt","children":list(objs)}; return i
    def MatLin(self,ksi,br):
        i=self._id(); self.materials[i]={"ksi":list(ksi),"br":float(br)}; return i
    def MatApl(self,obj,mat):
        self.obj_material[obj]=mat; self.applied.append((obj,mat)); return obj
    def ObjDivMag(self,obj,seg):
        self.divided.append((obj,seg)); return obj
    def RlxPre(self,obj): return 999
    def RlxAuto(self,intr,prec,maxiter,meth):
        self.relaxed=True; return [prec*2.0,1.2,0.3,maxiter] if self.fail_relax else [prec*0.1,1.2,0.3,12]
    def _magnitudes(self,obj):
        o=self.objects[obj]
        if o["type"]=="cnt":
            out=[]
            for child in o["children"]: out.extend(self._magnitudes(child))
            return out
        if obj in self.obj_material:
            return [self.materials[self.obj_material[obj]]["br"]]
        m=o["m"]; return [math.sqrt(sum(float(v)*float(v) for v in m))]
    def Fld(self,obj,comp,p):
        x,y,z=p
        mags=self._magnitudes(obj)
        amp=sum(mags)/len(mags) if mags else 1.0
        return [0.08*amp*math.sin(2*math.pi*z/50.0+math.pi/2),
                0.12*amp*math.sin(2*math.pi*z/50.0),0.002*amp*math.sin(z/17.0)]
    def UtiDelAll(self):
        self.next_id=1; self.objects={}; self.materials={}; self.obj_material={}
        self.applied=[]; self.divided=[]; self.relaxed=False
