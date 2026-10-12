"""Predicate-backed partial orders; unknown initial conditions stay explicit."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json


def validate_predicate(p):
    if not isinstance(p,dict):raise ValueError('initial-state predicates must be objects')
    if p.get('operator','equals')!='equals':raise ValueError('only equality predicates are supported')
    kind=p.get('type')
    fields=['source','relation','target'] if kind=='edge' else ['node' if kind=='node_attribute' else 'object','attribute']
    if kind not in {'edge','node_attribute','attribute'}:raise ValueError('unsupported predicate type')
    if not all(isinstance(p.get(f),str) and p[f] for f in fields):raise ValueError('predicate requires nonempty entity/relation/attribute strings')
    if kind=='edge' and type(p.get('present',True)) is not bool:raise ValueError('edge present must be boolean; omit unknown facts')
    if kind!='edge':
        if 'value' not in p or p['value'] is None:raise ValueError('attribute value is required; omit unknown facts')
        if p['attribute'] in {'is_open','is_on','gripper_empty'} and type(p['value']) is not bool:raise ValueError('state flag value must be boolean')


def predicate_key(p):
    if p.get('type')=='edge':return ('edge',p['source'],p['relation'],p['target'],bool(p.get('present',True)))
    if p.get('type') in {'node_attribute','attribute'}:
        return ('attribute',p.get('node',p.get('object')),p['attribute'],json.dumps(p['value'],sort_keys=True))
    return None


def effect_predicate(e):
    if e.get('operation') in {'add_edge','delete_edge'}:
        return {'type':'edge',**e['edge'],'present':e['operation']=='add_edge'}
    if e.get('operation')=='set_attribute':
        return {'type':'node_attribute','node':e['node'],'attribute':e['attribute'],'value':e['after']}
    return None


def opposite(a,b):return a is not None and b is not None and a[:-1]==b[:-1] and a[-1]!=b[-1]


def invalidates(action,key):
    for fx in action.get('effects',[]):
        p=effect_predicate(fx)
        if p and opposite(predicate_key(p),key):return True
        if fx.get('operation')=='delete_edges_matching' and key[0]=='edge' and key[-1]:
            pattern=fx.get('pattern',{})
            if all(pattern.get(field,value)==value for field,value in zip(['source','relation','target'],key[1:4])):return True
    # The old graph does not enumerate deletes for every prior location. Do not
    # silently carry a previous direct support/location across a move or pickup.
    if key[0]=='edge' and key[1]!='agent' and key[2] in {'ON','INSIDE'} and key[-1]:
        if action['action']=='PICK' and action.get('bindings',{}).get('$object')==key[1]:return True
        if action['action']=='PLACE' and action.get('bindings',{}).get('$object')==key[1]:
            return not any(predicate_key(effect_predicate(fx))==key for fx in action.get('effects',[]) if effect_predicate(fx))
    return False


def reaches(source,target,edges):
    successors=defaultdict(set)
    for e in edges:successors[e['source']].add(e['target'])
    todo=[source];seen=set()
    while todo:
        n=todo.pop()
        if n in seen:continue
        seen.add(n)
        if n==target:return True
        todo.extend(successors[n]-seen)
    return False


def order_summary(nodes,edges):
    ids={n['id'] for n in nodes};succ=defaultdict(set);incoming={n:0 for n in ids}
    for e in edges:
        if e['target'] not in succ[e['source']]:succ[e['source']].add(e['target']);incoming[e['target']]+=1
    layers=[];ready=sorted(n for n in ids if incoming[n]==0);seen=set()
    while ready:
        layers.append(ready);next_layer=[]
        for n in ready:
            seen.add(n)
            for t in succ[n]:
                incoming[t]-=1
                if incoming[t]==0:next_layer.append(t)
        ready=sorted(next_layer)
    # Strongly connected components distinguish cycles from downstream victims.
    indices={};low={};stack=[];on_stack=set();cycles=[]
    def visit(n):
        indices[n]=low[n]=len(indices);stack.append(n);on_stack.add(n)
        for t in succ[n]:
            if t not in indices:visit(t);low[n]=min(low[n],low[t])
            elif t in on_stack:low[n]=min(low[n],indices[t])
        if low[n]==indices[n]:
            component=[]
            while True:
                t=stack.pop();on_stack.remove(t);component.append(t)
                if t==n:break
            if len(component)>1 or n in succ[n]:cycles.append(sorted(component))
    for n in sorted(ids):
        if n not in indices:visit(n)
    holders=[n['id'] for n in nodes if n['action'] in {'PICK','PLACE','TILT','STIR','SCRUB'}]
    resources=[]
    if len(holders)>1:resources.append({'resource':'agent.gripper','capacity':1,'action_uses':holders,
        'basis':'shared_single_gripper_contract','ordering_required':'choose a compatible serialization; relative order is not inferred from semantic rank'})
    return {'acyclic':not cycles,'layers':layers,'cycles':cycles,'blocked_by_cycles':sorted(ids-seen),
            'layers_are_parallel_schedule':False,'resource_constraints':resources}


def mermaid(graph):
    names={n['id']:'n'+str(i) for i,n in enumerate(graph['nodes'])}
    lines=['flowchart TD']
    for n in graph['nodes']:
        label=n['api_call'].replace('&','&amp;').replace('"','&quot;').replace('<','&lt;').replace('>','&gt;')
        lines.append('  '+names[n['id']]+'["'+label+'"]')
    for e in graph['edges']:
        arrow='-.->' if e.get('depends_on_initial_state') else '-->'
        label='contract' if e['basis']=='declared_effect_condition' else 'text order'
        lines.append('  '+names[e['source']]+' '+arrow+'|"'+label+'"| '+names[e['target']])
    return '\n'.join(lines)


class DependencyBuilder:
    def __init__(self,mechanisms,scores=None,max_nodes=32):
        self.mechanisms=mechanisms;self.scores=scores or {};self.max_nodes=max_nodes
        self.producers=defaultdict(list)
        for m in mechanisms:
            for e in m.get('effects',[]):
                p=effect_predicate(e);key=predicate_key(p) if p else None
                if key:self.producers[key].append(m)

    def build(self,goals,initial_state=None,temporal=None,source='goal'):
        initial={};nodes={};edges=[];unresolved=[];unresolved_goals=[];roots=defaultdict(list);alternatives=[]
        for p in initial_state or []:
            validate_predicate(p)
            key=predicate_key(p)
            if key is None:raise ValueError('Unsupported initial-state predicate')
            if any(opposite(key,k) for k in initial):raise ValueError('Contradictory initial-state predicates')
            initial[key]=p
        edge_ids=set();satisfied=[]
        def link(a,b,p,basis,condition_basis=None):
            key=(a,b,json.dumps(p,sort_keys=True),basis)
            if key in edge_ids:return
            edge_ids.add(key);edges.append({'source':a,'target':b,'type':'precedes','predicate':p,
                'basis':basis,'condition_basis':condition_basis,
                'depends_on_initial_state':basis=='declared_effect_condition' and predicate_key(p) not in initial and
                                           not any(opposite(predicate_key(p),k) for k in initial),
                'causal_necessity_verified':False})
        def add_use(m,phase):
            id='use_'+hashlib.sha256((m['id']+'|'+str(phase)).encode()).hexdigest()[:16]
            if id not in nodes:
                if len(nodes)>=self.max_nodes:return None
                fields=['action','api_call','bindings','conditions','effects','role_types','role_type_constraints',
                        'parent_template_id','parent_generic_id','instantiation_basis','query_context_requirements','new_type_outcome_observed']
                n={k:deepcopy(m[k]) for k in fields if k in m}
                for scoped in m.get('query_phase_conditions',[]):
                    if scoped['from_phase']<=phase<=scoped['through_phase']:
                        n.setdefault('conditions',[]).append(deepcopy(scoped['condition']))
                evidence=m.get('evidence_statistics',{})
                n.update({'id':id,'type':'action','mechanism_ref':m['id'],'phase':phase,'sources':[source],
                          'evidence':{'ids':m.get('evidence_ids',[])[:12],'total_segments':len(m.get('evidence_ids',[])),
                                      'independent_trajectories':evidence.get('trajectory_count'),'tasks':evidence.get('task_count')},
                          'execution_verified':False})
                nodes[id]=n
            return id
        def existing_producer(key,phase,ancestors):
            eligible=[]
            for n in nodes.values():
                if n['phase']>phase or n['id'] in ancestors:continue
                if not any(predicate_key(effect_predicate(e))==key for e in n['effects'] if effect_predicate(e)):continue
                invalidated=any(other['phase']>=n['phase'] and other['phase']<phase and
                    other['id']!=n['id'] and invalidates(other,key)
                    for other in nodes.values())
                if not invalidated:eligible.append(n)
            return max(eligible,key=lambda n:n['phase'])['id'] if eligible else None
        def satisfy(p,phase,ancestors,consumer=None,depth=0,condition_basis=None,requested_action=None):
            key=predicate_key(p)
            if key is None:
                unresolved.append({'consumer':consumer,'predicate':p,'reason':'unsupported predicate'});return None
            # Initial values can be consumed directly only if no earlier phase
            # contradicts them; this prevents reusing a stale held-object state.
            stale=any(n['phase']<phase and invalidates(n,key) for n in nodes.values())
            if key in initial and not stale:
                satisfied.append({'consumer':consumer,'predicate':p,'basis':initial[key].get('basis','provided_initial_state')});return None
            empty_condition=key[0]=='attribute' and key[1]=='agent' and key[2]=='gripper_empty'
            if consumer and empty_condition:
                unresolved.append({'consumer':consumer,'predicate':p,'reason':'gripper availability requires initial state or resource serialization; no arbitrary release order inferred'})
                return None
            previous=existing_producer(key,phase,ancestors)
            if previous:
                if consumer:link(previous,consumer,p,'declared_effect_condition',condition_basis)
                return previous
            # These conditions require initial scene/resource information.
            # Recursing into arbitrary PLACE to empty the hand creates cycles.
            scene_condition=key[0]=='edge' and key[1]=='agent' and key[2] in {'NEAR','AT_WORK_AREA'}
            known_false=any(opposite(key,k) for k in initial)
            if consumer and (scene_condition or empty_condition) and not known_false:
                unresolved.append({'consumer':consumer,'predicate':p,'reason':'initial scene/resource state unspecified',
                                   'candidate_producers':[m['api_call'] for m in self.producers.get(key,[])][:5]});return None
            choices=[m for m in self.producers.get(key,[]) if not any(nodes[n]['mechanism_ref']==m['id'] and nodes[n]['phase']==phase for n in ancestors)]
            if requested_action:choices=[m for m in choices if m['action']==requested_action]
            if not choices or depth>=10:
                record={'consumer':consumer,'predicate':p,'reason':'no effect-compatible producer' if not choices else 'dependency depth limit'}
                (unresolved if consumer else unresolved_goals).append(record);return None
            choices.sort(key=lambda m:(self.scores.get(m['id'],0),-len(m.get('bindings',{}))),reverse=True)
            m=choices[0]
            if len(choices)>1:alternatives.append({'predicate':p,'chosen':m['id'],
                'candidates':[{'mechanism_ref':x['id'],'api_call':x['api_call'],'score':self.scores.get(x['id'],0)} for x in choices[:5]],
                'basis':'effect match; semantic score breaks ambiguity' if self.scores else 'effect match; deterministic candidate choice',
                'binding_verified':False})
            id=add_use(m,phase)
            if id is None:
                (unresolved if consumer else unresolved_goals).append({'consumer':consumer,'predicate':p,'reason':'action-node budget'});return None
            for fx in m.get('effects',[]):
                if fx.get('applicability_condition') and predicate_key(effect_predicate(fx))==key:
                    guard=deepcopy(fx['applicability_condition'])
                    nodes[id]['conditions'].append({'predicate':guard,'basis':'recorded_transfer_content_membership_guard'})
            for cond in nodes[id].get('conditions',[]):satisfy(cond['predicate'],phase,ancestors|{id},id,depth+1,cond.get('basis'))
            if consumer:link(id,consumer,p,'declared_effect_condition',condition_basis)
            return id
        for goal in sorted(goals,key=lambda g:g.get('phase',0)):
            root=satisfy(goal['predicate'],goal.get('phase',0),set(),requested_action=goal.get('requested_action'))
            if root:roots[goal.get('phase',0)].append(root)
        # Text requests constrain phase roots and the next phase's entry actions.
        for t in temporal or []:
            first=list(dict.fromkeys(roots[t['before']]))
            phase_nodes={n['id'] for n in nodes.values() if n['phase']==t['after']}
            entries=phase_nodes-{e['target'] for e in edges if e['source'] in phase_nodes and e['target'] in phase_nodes}
            for a in first:
                for b in entries:
                    if a!=b:link(a,b,{'type':'requested_order','before_phase':t['before'],'after_phase':t['after']},t.get('basis','explicit_text_order'))
        graph={'nodes':list(nodes.values()),'edges':edges,'goal_states':deepcopy(goals),'alternatives':alternatives,
               'unresolved_conditions':unresolved,'unresolved_goals':unresolved_goals,'initial_satisfied':satisfied,
               'order':order_summary(list(nodes.values()),edges),'route':source,
               'complete_execution_plan':False,'dependency_scope':'declared contracts and explicit text; initial state and bindings may remain unknown'}
        graph['state_threats']=[]
        final_held={g['predicate']['target'] for g in goals if g.get('scope')=='final' and g['predicate'].get('type')=='edge' and
                    g['predicate'].get('source')=='agent' and g['predicate'].get('relation')=='HOLDS' and g['predicate'].get('present',True)}
        graph['resource_conflicts']=([{'resource':'agent.gripper','capacity':1,'final_held_objects':sorted(final_held),
            'reason':'more than one final held object cannot fit the declared single gripper'}] if len(final_held)>1 else [])
        for e in edges:
            if e['basis']!='declared_effect_condition':continue
            key=predicate_key(e['predicate'])
            for n in nodes.values():
                if n['id'] in {e['source'],e['target']}:continue
                if invalidates(n,key):
                    if not reaches(n['id'],e['source'],edges) and not reaches(e['target'],n['id'],edges):
                        graph['state_threats'].append({'action':n['id'],'dependency':[e['source'],e['target']],
                                                     'predicate':e['predicate'],'reason':'may invalidate state between producer and consumer; scheduling unresolved'})
        graph['mermaid']=mermaid(graph)
        graph['support_status']=('conflict' if not graph['order']['acyclic'] or graph['state_threats'] or graph['resource_conflicts'] else
                                 'unresolved' if unresolved or unresolved_goals else
                                 'conditional' if alternatives or graph['order']['resource_constraints'] or any(e.get('depends_on_initial_state') for e in edges) else 'complete')
        graph['complete_execution_plan']=False
        return graph
