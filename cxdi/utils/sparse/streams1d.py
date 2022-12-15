""" 
Module to convert images to pixel,time_of_arrival,nphot arrays
Each frame is flatten to 1d array to have a single index
"""
import numpy as np
from collections import namedtuple

sparse_stream = namedtuple("sparse_stream",["pixel","time_of_arrival","nphot"])
__all__ = ["images_to_sparse_stream","concatenate_sparse_streams","save_stream","load_stream","sparse_stream_to_images","ravel","unravel"]


def ravel(imgs):
    return imgs.reshape( (imgs.shape[0],-1) )

def unravel(imgs,shape):
    return imgs.reshape( (-1,)+shape )

def compare_streams(s1,s2):
    return all([ np.alltrue(getattr(s1,k) == getattr(s2,k)) for k in s1._fields])

def images_to_sparse_stream(imgs,dt=1,t0=0):
    """ return positions, arrival time and "time over threshold"
        (i.e. number of photons)
        """
    imgs = imgs.reshape( (imgs.shape[0],-1) )
    t,pixel=np.nonzero(imgs>0)
    nphotons = imgs[t,pixel]
    if dt != 1: t *= dt
    if t0 != 0: t += t0
    return sparse_stream(pixel=pixel,time_of_arrival=t,nphot=nphotons)

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

def sparse_stream_to_images(stream,frame_shape,dt=1,nmax=None,weight_tot=True):
    """ stream is tuple-like data index,time_of_arrival,nphot
        or an array (nevents,3)
        if weight_tot is True, it is used as number of photons
    """
    stream = np.asarray(stream)
    if stream.shape[0] != 3: stream = stream.T
    p,t,n = stream
    if nmax is None:
        nmax = t.max()/dt
    n0 = t.min()/dt
    tbins = np.arange(n0,nmax+2)*dt-dt/2
    tbins = np.arange(n0,nmax+1)*dt
    idx = np.digitize(t,tbins,right=True)
    imgs = np.zeros( (len(tbins),frame_shape[0]*frame_shape[1]),dtype=int )
    if weight_tot:
        imgs[idx,p] += n
    else:
        imgs[idx,p] += 1
    return unravel(imgs,frame_shape)


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
