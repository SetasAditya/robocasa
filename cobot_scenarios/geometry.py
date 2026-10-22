"""Conservative base-footprint screen; not a manipulation/controller success test."""
import math
from collections import deque

def fitting_yaw(size, region, requested, clearance=.002):
    """Nearest feasible yaw in radians, retaining the requested yaw when it fits.

    Use the same conservative rectangular object bounds as RoboCasa. Never
    shrink the object or relax the placement region to admit a requested pose.
    """
    for delta in [0]+[sign*math.radians(d) for d in range(1,181) for sign in (1,-1)]:
        angle=requested+delta;c=abs(math.cos(angle));s=abs(math.sin(angle))
        if c*size[0]+s*size[1]+clearance<=region[0] and s*size[0]+c*size[1]+clearance<=region[1]:
            return angle
    raise ValueError('Object cannot fit placement region at any sampled yaw')

def route(start,goal,rectangles,bounds,radius=.45,resolution=.15,goal_snap=0.):
    """Axis-aligned fixture bounds inflated by CoBot radius. Returns grid path or None."""
    xmin,ymin,xmax,ymax=bounds
    nx=int((xmax-xmin)/resolution)+1;ny=int((ymax-ymin)/resolution)+1
    if nx*ny>500000:raise ValueError('Unexpected scene bounds')
    def point(c):return (xmin+c[0]*resolution,ymin+c[1]*resolution)
    def cell(p):return (round((p[0]-xmin)/resolution),round((p[1]-ymin)/resolution))
    def free(c):
        x,y=point(c)
        return 0<=c[0]<nx and 0<=c[1]<ny and not any(a-radius<=x<=c_+radius and b-radius<=y<=d+radius for a,b,c_,d in rectangles)
    source=cell(start);target=cell(goal)
    if not free(source):return None
    if not free(target) and goal_snap:
        candidates=[(i,j) for i in range(nx) for j in range(ny) if math.dist(point((i,j)),goal)<=goal_snap and free((i,j))]
        if candidates:target=min(candidates,key=lambda c:math.dist(point(c),goal))
    if not free(target):return None
    parents={source:None};queue=deque([source])
    while queue:
        c=queue.popleft()
        if c==target:
            path=[]
            while c is not None:path.append(point(c));c=parents[c]
            return path[::-1]
        for d in ((1,0),(-1,0),(0,1),(0,-1)):
            n=(c[0]+d[0],c[1]+d[1])
            if n not in parents and free(n):parents[n]=c;queue.append(n)
    return None

def fixture_bounds(env):
    import numpy as np
    rects=[]
    for f in env.fixtures.values():
        # Ignore above-base shelves and visual floors; include walls/base cabinets.
        if any(k in type(f).__name__.lower() for k in ('floor','accessory','wallaccessory')):continue
        if type(f).__name__=='Wall':
            for gid in range(env.sim.model.ngeom):
                name=env.sim.model.geom_id2name(gid) or ''
                if not name.startswith(f.name+'_') or env.sim.model.geom_type[gid]!=6:continue
                center=env.sim.data.geom_xpos[gid]
                extent=np.abs(env.sim.data.geom_xmat[gid].reshape(3,3))@env.sim.model.geom_size[gid]
                if center[2]-extent[2]>.65:continue
                rects.append((float(center[0]-extent[0]),float(center[1]-extent[1]),float(center[0]+extent[0]),float(center[1]+extent[1])))
            continue
        p=np.asarray(f.pos);size=np.asarray(f.size)
        if p[2]-size[2]/2>.65 or p[2]+size[2]/2<.05:continue
        angle=float(getattr(f,'rot',0.));ext=np.abs(np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]))@size[:2]/2
        rects.append((float(p[0]-ext[0]),float(p[1]-ext[1]),float(p[0]+ext[0]),float(p[1]+ext[1])))
    return rects
