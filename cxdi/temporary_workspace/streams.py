import numpy as np
from collections import namedtuple

_stream = namedtuple("stream",["x","y","time_of_arrival","time_over_threshold"])

def images_to_stream(imgs,dt=1):
    """ return positions, arrival time and "time over threshold"
        (i.e. number of photons)
        """
    t,x,y=np.nonzero(imgs>0)
    nphotons = imgs[t,x,y]
    if dt != 1: t = t*dt
    return _stream(x=x,y=y,time_of_arrival=t,time_over_threshold=nphotons)

def stream_to_images(stream,img_shape="auto",dt=1,nmax=None,weight_tot=True):
    """ stream is tuple-like data x,y,time_of_arrival,time_over_threshold 
        or an array (nevents,4)
        if img_shape is "auto", the max of x and y is used. For low avg counts
        it might be better to force the shape (low probability of finding hits
        up to the edges
        if weight_tot is True, it is used as number of photons
    """
    stream = np.asarray(stream)
    if stream.shape[0] != 4: stream = stream.T
    x,y,t,n = stream
    if nmax is None:
        nmax = t.max()/dt
    n0 = t.min()/dt
    tbins = np.arange(n0,nmax+2)*dt-dt/2
    tbins = np.arange(n0,nmax+1)*dt
    idx = np.digitize(t,tbins,right=True)
    if isinstance(img_shape,str) and img_shape == "auto":
        img_shape = x.max()+1,y.max()+1 # +1 because starts at zero
    imgs = np.zeros( (len(tbins),)+img_shape,dtype=int )
    if weight_tot:
        imgs[idx,x,y] += n
    else:
        imgs[idx,x,y] += 1
    return imgs
