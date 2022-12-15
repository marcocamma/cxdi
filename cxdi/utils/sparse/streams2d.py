""" 
Module to convert images to x,y,time_of_arrival,nphot arrays
"""
import numpy as np
from collections import namedtuple

sparse_stream = namedtuple("sparse_stream",["x","y","time_of_arrival","nphot"])
__all__ = ["images_to_sparse_stream","concatenate_sparse_streams","save_stream","load_stream","sparse_stream_to_images"]


def compare_streams(s1,s2):
    return all([ np.alltrue(getattr(s1,k) == getattr(s2,k)) for k in s1._fields])

def images_to_sparse_stream(imgs,dt=1,t0=0):
    """ return positions, arrival time and "time over threshold"
        (i.e. number of photons)
        """
    t,y,x=np.nonzero(imgs>0)
    nphotons = imgs[t,y,x]
    if dt != 1: t *= dt
    if t0 != 0: t += t0
    return sparse_stream(x=x,y=y,time_of_arrival=t,nphot=nphotons)

def concatenate_sparse_streams(*streams):
    if len(streams) == 1 and not isinstance(streams[0],sparse_stream):
        streams = streams[0]
    if len(streams) == 0:
        raise ValueError("No sparse_streams provided")
    elif len(streams) == 1 and isinstance(streams[0],sparse_stream):
        return streams[0]
    else:
        ret = dict()
        for key in streams[0]._fields:
            ret[key] = np.concatenate([ getattr(s,key) for s in streams])
        ret = sparse_stream(*ret.values())
    return ret


def save_stream(stream,filename):
    tosave = stream._asdict()
    np.save(filename,tosave)

def load_stream(filename):
    data = np.load(filename,allow_pickle=True).item()
    return sparse_stream(*data.values())

def sparse_stream_to_images(stream,frame_shape="auto",dt=1,nmax=None,weight_tot=True):
    """ stream is tuple-like data x,y,time_of_arrival,nphot
        or an array (nevents,4)
        if frame_shape is "auto", the max of x and y is used. For low avg counts
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
    if isinstance(frame_shape,str) and frame_shape == "auto":
        frame_shape = x.max()+1,y.max()+1 # +1 because starts at zero
    imgs = np.zeros( (len(tbins),)+frame_shape,dtype=int )
    if weight_tot:
        imgs[idx,y,x] += n
    else:
        imgs[idx,y,x] += 1
    return imgs


def test1(shape=(200,128,256)):
    """ test imgs->stream->imgs """
    imgs = np.random.randint(0,high=10,size=shape)
    stream = images_to_sparse_stream(imgs)
    imgs2 = sparse_stream_to_images(stream,frame_shape=imgs.shape[1:])
    assert np.alltrue(imgs==imgs2)

def test2(shape=(200,128,256)):
    """ test concatenate_sparse_streams arguments """
    imgs = np.random.randint(0,high=10,size=shape)
    n = shape[0]//3
    stream1 = images_to_sparse_stream(imgs[:n])
    stream2 = images_to_sparse_stream(imgs[n:],t0=n)
    stream_c1 = concatenate_sparse_streams(stream1,stream2)
    stream_c2 = concatenate_sparse_streams( (stream1,stream2) )
    assert compare_streams(stream_c1,stream_c2)

def test3(shape=(200,128,256)):
    """ test concatenate_sparse_streams """
    imgs = np.random.randint(0,high=10,size=shape)
    n = shape[0]//3
    stream1 = images_to_sparse_stream(imgs[:n])
    stream2 = images_to_sparse_stream(imgs[n:],t0=n)
    stream = concatenate_sparse_streams(stream1,stream2)
    imgs2 = sparse_stream_to_images(stream,frame_shape=imgs.shape[1:])
    assert np.alltrue(imgs==imgs2)

def test4(shape=(200,128,256)):
    """ test save/reload """
    imgs = np.random.randint(0,high=10,size=shape)
    stream = images_to_sparse_stream(imgs)
    save_stream(stream,"/tmp/sparse_saving_text.npy")
    stream2 = load_stream("/tmp/sparse_saving_text.npy")
    assert compare_streams(stream,stream2)



def test_all():
    keys = globals()
    for k in keys:
        f = globals()[k]
        if callable(f) and k.find("test") == 0 and k != "test_all":
            print("testing",k)
            f()
