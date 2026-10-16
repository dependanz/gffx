"""One eager camera composition; backend objects supply arrays and native transforms.

This module contains no framework import. A binding supplies array creation/layout
and validation, not a second projection implementation. No compile/export claim.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple, Any


class ProjectedPoints(NamedTuple):
    pixels: Any
    ndc_depth: Any
    view_depth: Any
    valid: Any


class Camera:
    _backend = None
    distortion = None
    exposure = 1.0
    rolling_shutter = None
    lens_stack = None
    projection_kind = "perspective"

    @classmethod
    def from_intrinsics(cls, fx, fy, cx, cy, image_height, image_width, near, far, pose=None):
        b=cls._backend; xp=b.xp
        for name,value in [('image_height',image_height),('image_width',image_width)]:
            if type(value) is not int or value<=0: raise ValueError(name+' must be a positive integer')
        if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in (near,far)):
            raise ValueError('near and far must be finite static real numbers')
        if not 0<near<far: raise ValueError('require 0 < near < far')
        tensors=[v for v in (fx,fy,cx,cy,pose) if b.is_array(v)]
        like=tensors[0] if tensors else None
        for value in tensors: b.check_float(value,like)
        values=[b.intrinsic(v,like) for v in (fx,fy,cx,cy)]
        if any(v.ndim!=1 or v.shape[0]<1 for v in values): raise ValueError('intrinsics must be scalar or nonempty [B]')
        batch=max(v.shape[0] for v in values)
        # Only Python numbers broadcast at construction, never a mismatched tensor batch.
        values=[b.expand_scalar(v,batch) if not b.is_array(raw) else v for raw,v in zip((fx,fy,cx,cy),values)]
        if any(v.shape[0]!=batch for v in values): raise ValueError('intrinsic batch sizes must agree')
        if any(not bool(xp.all(xp.isfinite(v))) for v in values): raise ValueError('intrinsics must be finite')
        if not bool(xp.all(values[0]>0)) or not bool(xp.all(values[1]>0)): raise ValueError('focal lengths must be positive')
        if pose is None: pose=b.identity(batch,values[0])
        else:
            b.check_float(pose,values[0])
            if tuple(pose.shape)!=(batch,4,4): raise ValueError('pose must have shape [B,4,4] matching intrinsics')
            if not bool(xp.all(xp.isfinite(pose))): raise ValueError('pose must be finite')
            b.inverse(pose)  # eager singularity check; do not cache a detached inverse
        self=cls();self.fx,self.fy,self.cx,self.cy=values;self.pose=pose
        self.image_height=image_height;self.image_width=image_width
        self.near=float(near);self.far=float(far);self.batch_size=batch
        return self

    @classmethod
    def from_orthographic(cls, fx, fy, cx, cy, image_height, image_width, near, far, pose=None):
        """Parallel rays: fx/fy are positive pixels per view-space unit.

        u=fx*x_view+cx, v=-fy*y_view+cy. Pose must be affine and invertible.
        """
        self=cls.from_intrinsics(fx,fy,cx,cy,image_height,image_width,near,far,pose)
        b=cls._backend;xp=b.xp
        if (not bool(xp.all(self.pose[:,3,:3]==0)) or
                not bool(xp.all(self.pose[:,3,3]==1))):
            raise ValueError('orthographic pose must be affine with last row [0,0,0,1]')
        self.projection_kind='orthographic'
        return self

    @classmethod
    def from_weak_perspective(cls, scale, tx, ty, image_height, image_width, near, far, pose=None):
        """Normalized image coordinates s*(x+tx), -s*(y+ty), x right/y down.

        This is parallel projection with scale, not a perspective approximation.
        Depth stays geometric -z_view; scale does not rescale clipping planes.
        """
        c=cls.from_orthographic(scale,scale,tx,ty,image_height,image_width,near,far,pose)
        fx=c.fx*image_width/2;fy=c.fy*image_height/2
        cx=(1+c.fx*c.cx)*image_width/2;cy=(1-c.fy*c.cy)*image_height/2
        return cls.from_orthographic(fx,fy,cx,cy,image_height,image_width,near,far,c.pose)

    def projection(self):
        xp=self._backend.xp;z=xp.zeros_like(self.fx);one=xp.ones_like(self.fx)
        n,f=self.near,self.far
        # Tensor denominators retain division rather than a backend scalar
        # reciprocal-multiply shortcut (different last bits for widths like 3).
        width=one*self.image_width;height=one*self.image_height
        if self.projection_kind=='orthographic':
            return xp.stack((xp.stack((2*self.fx/width,z,z,2*self.cx/width-1),axis=-1),
                             xp.stack((z,2*self.fy/height,z,1-2*self.cy/height),axis=-1),
                             xp.stack((z,z,one*(2/(n-f)),one*((f+n)/(n-f))),axis=-1),
                             xp.stack((z,z,z,one),axis=-1)),axis=-2)
        return xp.stack((xp.stack((2*self.fx/width,z,1-2*self.cx/width,z),axis=-1),
                         xp.stack((z,2*self.fy/height,2*self.cy/height-1,z),axis=-1),
                         xp.stack((z,z,one*((f+n)/(n-f)),one*((2*f*n)/(n-f))),axis=-1),
                         xp.stack((z,z,-one,z),axis=-1)),axis=-2)

    def world_to_view(self): return self._backend.inverse(self.pose)

    def world_to_clip(self): return self.projection() @ self.world_to_view()

    def _pack(self, points, width, offsets=None):
        b=self._backend;b.check_float(points,self.fx,cast_dtype=True)
        if points.ndim==3:
            if offsets is not None: raise ValueError('offsets are only accepted with packed [P,C] points')
            if points.shape[0]!=self.batch_size or points.shape[-1]!=width: raise ValueError('dense points must have shape [B,N,%d]'%width)
            shape=tuple(points.shape[:-1]);offsets=b.offsets(self.batch_size,points.shape[1],points)
        elif points.ndim==2 and points.shape[-1]==width:
            shape=tuple(points.shape[:-1])
            if offsets is None:
                if self.batch_size!=1: raise ValueError('a rig requires [B,N,C] or explicit point_offsets')
                offsets=b.offsets(1,points.shape[0],points)
            else:b.check_offsets(offsets,points.shape[0],self.batch_size,points)
        else: raise ValueError('points must have shape [N,%d], [B,N,%d], or packed [P,%d]'%(width,width,width))
        return b.dense(points.reshape(-1,width)),offsets,shape

    def project(self, points, point_offsets=None, *, in_front_only=False):
        if type(in_front_only) is not bool: raise TypeError('in_front_only must be bool')
        b=self._backend;xp=b.xp;p,offsets,shape=self._pack(points,3,point_offsets)
        matrix=b.cast(self.world_to_clip(),p)
        clip=b.transform(p,b.dense(matrix),offsets)
        ndc,valid=b.divide(clip)
        pixels=xp.stack(((ndc[:,0]+1)*self.image_width/2,(1-ndc[:,1])*self.image_height/2),axis=-1)
        view=b.transform(p,b.dense(b.cast(self.world_to_view(),p)),offsets)
        depth=-view[:,2]
        if in_front_only:
            valid=valid & (depth>b.eps)
            pixels=xp.where(valid[:,None],pixels,xp.zeros_like(pixels))
            ndc=xp.where(valid[:,None],ndc,xp.zeros_like(ndc))
            depth=xp.where(valid,depth,xp.zeros_like(depth))
        return ProjectedPoints(pixels.reshape(*shape,2),ndc[:,2].reshape(shape),depth.reshape(shape),valid.reshape(shape))

    def view_depth_to_ndc(self, view_depth):
        self._backend.check_float(view_depth,self.fx,cast_dtype=True)
        n,f=self.near,self.far
        if self.projection_kind=="orthographic":return (2*view_depth-(f+n))/(f-n)
        return -(f+n)/(n-f)+(2*f*n)/(n-f)/view_depth

    def ndc_to_view_depth(self, ndc_depth):
        self._backend.check_float(ndc_depth,self.fx,cast_dtype=True)
        n,f=self.near,self.far
        if self.projection_kind=="orthographic":return (ndc_depth*(f-n)+(f+n))/2
        return ((2*f*n)/(n-f))/(ndc_depth+(f+n)/(n-f))

    def _unproject(self, pixels, depth, kind, mask=None):
        b=self._backend;xp=b.xp
        if kind not in ('view','ndc'): raise ValueError("kind must be 'view' or 'ndc'")
        b.check_float(depth,pixels)
        if tuple(depth.shape)!=tuple(pixels.shape[:-1]): raise ValueError('depth must match the pixel batch/point shape')
        good=xp.isfinite(depth) & xp.all(xp.isfinite(pixels),axis=-1)
        if mask is not None:
            b.check_mask(mask,depth);good=good & mask
        if kind=='view' and self.projection_kind=='perspective':good=good & (xp.abs(depth)>b.eps)
        safe_depth=xp.where(good,depth,xp.ones_like(depth) if kind=='view' else xp.zeros_like(depth))
        z=self.view_depth_to_ndc(safe_depth) if kind=='view' else safe_depth
        safe_pixels=xp.where(good[...,None],pixels,xp.zeros_like(pixels))
        ndc=xp.stack((safe_pixels[...,0]*2/self.image_width-1,1-safe_pixels[...,1]*2/self.image_height,z),axis=-1)
        p,offsets,shape=self._pack(ndc,3)
        inverse=b.dense(b.cast(b.inverse(self.world_to_clip()),p))
        h=b.transform(p,inverse,offsets)
        # The inverse homogeneous divide has the same guarded semantics as forward projection.
        xyz,divide_valid=b.divide(h)
        good=good & divide_valid.reshape(shape) & xp.all(xp.isfinite(xyz.reshape(*shape,3)),axis=-1)
        xyz=xp.where(good[...,None],xyz.reshape(*shape,3),xp.zeros_like(xyz.reshape(*shape,3)))
        return xyz,good

    def unproject_points(self, pixels, depth, *, kind):
        self._pack(pixels,2)  # shape/device validation before arithmetic
        return self._unproject(pixels,depth,kind)[0]

    def unproject(self, depth, *, kind, valid=None):
        b=self._backend;b.check_float(depth,self.fx,cast_dtype=True)
        if depth.ndim!=3 or depth.shape[0]!=self.batch_size: raise ValueError('depth must have shape [B,H,W]')
        # A subwindow uses the original top-left pixel coordinates; no implicit resizing.
        pixels=b.pixel_grid(depth)
        xyz,good=self._unproject(pixels.reshape(self.batch_size,-1,2),depth.reshape(self.batch_size,-1),kind,
            None if valid is None else b.checked_grid_mask(valid,depth).reshape(self.batch_size,-1))
        return xyz.reshape(*depth.shape,3),good.reshape(depth.shape)

    @classmethod
    def stack(cls,cameras):
        cameras=list(cameras)
        if not cameras: raise ValueError('Camera.stack requires at least one camera')
        first=cameras[0]
        if any(not isinstance(c,cls) for c in cameras): raise TypeError('all cameras must use the same backend')
        if any(c.projection_kind!=first.projection_kind for c in cameras):
            raise ValueError('stacked cameras must share projection kind')
        attrs=('image_height','image_width','near','far')
        if any(any(getattr(c,k)!=getattr(first,k) for k in attrs) for c in cameras):
            raise ValueError('stacked cameras must share image size and clipping planes')
        b=cls._backend
        for c in cameras:b.check_float(c.fx,first.fx)
        values={k:b.xp.concat([getattr(c,k) for c in cameras],axis=0) for k in ('fx','fy','cx','cy','pose')}
        constructor=cls.from_orthographic if first.projection_kind=='orthographic' else cls.from_intrinsics
        return constructor(**values,**{k:getattr(first,k) for k in attrs})
