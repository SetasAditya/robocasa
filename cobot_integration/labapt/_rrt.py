"""Deterministic joint-space RRT-Connect; no simulator or live state access.

collision_fn receives a fresh configuration copy and returns None/False when
free, or a truthy collision description. It must evaluate its independent
planning clone. Bounds apply to explicit joint winding; no angle wrapping occurs.
"""
import numpy as np

_MAX_STEP = .30
_EDGE_SPACING = .10


def plan(start, goals, low, high, collision_fn, seed=1729, iterations=2500):
    """Return start-to-one-goal configuration copies, or None if no path found.

    Each returned edge has Euclidean joint length <= .30 rad. Every edge is
    checked at endpoints and interpolation spacing <= .10 rad. The caller owns
    any execution/controller tracking checks; this function never executes.
    """
    start=np.asarray(start,dtype=float).copy()
    low=np.asarray(low,dtype=float).copy();high=np.asarray(high,dtype=float).copy()
    goals=np.asarray(goals,dtype=float).copy()
    if goals.size==0:return None
    if goals.ndim==1:goals=goals[None,:]
    if start.ndim!=1 or low.shape!=start.shape or high.shape!=start.shape or goals.ndim!=2 or goals.shape[1]!=start.size:
        raise ValueError('Start, bounds and goal dimensions must agree')
    if not all(np.isfinite(x).all() for x in (start,low,high,goals)) or np.any(high<=low):
        raise ValueError('Finite configurations and strictly ordered finite bounds required')
    if not isinstance(iterations,(int,np.integer)) or iterations<0:raise ValueError('iterations must be a nonnegative integer')
    def in_bounds(q):return bool(np.all(q>=low) and np.all(q<=high))
    def free(q):return not bool(collision_fn(q.copy()))
    if not in_bounds(start) or not free(start):return None
    valid=[]
    for goal in goals:
        if in_bounds(goal) and free(goal) and not any(np.array_equal(goal,g) for g in valid):valid.append(goal.copy())
    if not valid:return None
    def edge_free(a,b):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/_EDGE_SPACING)))
        return all(free(a+(b-a)*(j/count)) for j in range(count+1))
    def subdivide(a,b):
        count=max(1,int(np.ceil(np.linalg.norm(b-a)/_MAX_STEP)))
        return [a+(b-a)*(j/count) for j in range(count+1)]
    for goal in sorted(valid,key=lambda q:float(np.linalg.norm(q-start))):
        if np.array_equal(start,goal):return [start.copy()]
        if edge_free(start,goal):return subdivide(start,goal)
    rng=np.random.default_rng(seed)
    # The goal tree is a forest: each collision-free goal is an independent root.
    a=([start.copy()],[-1]);b=([g.copy() for g in valid],[-1]*len(valid));a_is_start=True
    def extend(tree,target):
        nodes,parents=tree
        nearest=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)))
        delta=target-nodes[nearest];distance=float(np.linalg.norm(delta))
        if distance<1e-12:return 'reached',nearest
        candidate=nodes[nearest]+delta*min(1.,_MAX_STEP/distance)
        if not edge_free(nodes[nearest],candidate):return 'trapped',nearest
        nodes.append(candidate.copy());parents.append(nearest)
        return ('reached' if distance<=_MAX_STEP else 'advanced'),len(nodes)-1
    def trace(tree,index):
        nodes,parents=tree;path=[]
        while index>=0:path.append(nodes[index].copy());index=parents[index]
        return path[::-1]
    for _ in range(iterations):
        goal=valid[int(rng.integers(len(valid)))];choice=rng.random()
        if choice<.15:
            sample=(goal if a_is_start else start).copy()
        elif choice<.55:
            sample=rng.uniform(low,high)
        else:
            center=start+rng.random()*(goal-start)
            sample=np.clip(center+rng.normal(size=start.size)*np.minimum(1.,(high-low)*.18),low,high)
        status,ia=extend(a,sample)
        if status!='trapped':
            # Greedily connect the other tree to the newly added configuration.
            while True:
                status,ib=extend(b,a[0][ia])
                if status!='advanced':break
            if status=='reached':
                pa=trace(a,ia);pb=trace(b,ib)
                result=pa+pb[-2::-1] if a_is_start else pb+pa[-2::-1]
                return [q.copy() for q in result]
        a,b=b,a;a_is_start=not a_is_start
    return None
