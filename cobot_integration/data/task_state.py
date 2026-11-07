"""Pickle-free native task history; unknown stateful objects require an adapter."""
import ast
import inspect
import textwrap
from functools import lru_cache
import numpy as np


def pack(value):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return {'$array':value.tolist(),'dtype':str(value.dtype)}
    if isinstance(value,tuple):return {'$tuple':[pack(v) for v in value]}
    if isinstance(value,set):return {'$set':[pack(v) for v in sorted(value,key=repr)]}
    if isinstance(value,list):return [pack(v) for v in value]
    if isinstance(value,dict) and all(isinstance(k,str) for k in value):return {k:pack(v) for k,v in value.items()}
    raise TypeError('Task history needs a state adapter: '+type(value).__name__)


def unpack(value):
    if isinstance(value,list):return [unpack(v) for v in value]
    if isinstance(value,dict):
        if '$array' in value:return np.asarray(value['$array'],dtype=value['dtype'])
        if '$tuple' in value:return tuple(unpack(v) for v in value['$tuple'])
        if '$set' in value:return set(unpack(v) for v in value['$set'])
        return {k:unpack(v) for k,v in value.items()}
    return value


@lru_cache(None)
def declared_fields(kind):
    names=set()
    for cls in kind.__mro__:
        if cls.__name__=='Kitchen':break
        try:tree=ast.parse(textwrap.dedent(inspect.getsource(cls)))
        except (OSError,TypeError) as exc:raise TypeError('Task class needs an explicit state adapter') from exc
        for node in ast.walk(tree):
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and isinstance(node.ctx,ast.Store):
                names.add(node.attr)
    return tuple(sorted(names))


def capture(env):
    if hasattr(env,'challenge_state'):return {'adapter':'challenge','value':env.challenge_state()}
    from robosuite.models.objects import MujocoObject
    from robocasa.models.fixtures import Fixture
    saved={}
    for name in declared_fields(type(env)):
        if name not in vars(env):continue
        value=vars(env)[name]
        if isinstance(value,(Fixture,MujocoObject)):continue  # Model references rebuilt from ep_meta, not temporal state.
        saved[name]=pack(value)
    return {'adapter':'native_declared_fields','value':saved}


def restore(env,state):
    if state['adapter']=='challenge':env.restore_challenge_state(state['value']);return
    if state['adapter']!='native_declared_fields':raise ValueError('Unknown task-state adapter')
    allowed=set(declared_fields(type(env)))
    for name,value in state['value'].items():
        if name not in allowed:raise ValueError('Unexpected native task history field')
        setattr(env,name,unpack(value))
