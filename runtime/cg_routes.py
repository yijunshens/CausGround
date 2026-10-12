"""Two diagnostic routes and one goal-relevant, semantically ranked closure."""
from copy import deepcopy
from collections import defaultdict
from cg_dependency import DependencyBuilder, effect_predicate, predicate_key, order_summary, mermaid
from cg_goals import extract_goals
from cg_text import mentioned
import re


def constrained_pool(pool,extraction):
    excluded=[]
    for m in pool:
        for prohibition in extraction['prohibitions']:
            text=prohibition['text'];action=m['action'];applies=False
            if re.search(r'\b(?:start|turn|switch|activate)\b',text):
                desired=not bool(re.search(r'\boff\b',text))
                applies=any(fx.get('attribute')=='is_on' and fx.get('after')==desired for fx in m.get('effects',[]))
            else:
                verbs={'OPEN':'open','CLOSE':'close','PICK':'pick|take|grab|retrieve','PLACE':'place|put',
                       'STIR':'stir','SCRUB':'scrub|clean','TILT':'tilt|pour','WAIT':'wait','SLIDE':'slide'}
                applies=bool(action in verbs and re.search(r'\b(?:'+verbs.get(action,'^$')+r')\b',text))
            if not applies:continue
            values=[v for v in m['bindings'].values() if v not in {'on','off','ON','IN','NEAR'} and not v.startswith('$')]
            all_values=[v for candidate in pool for v in candidate['bindings'].values() if not v.startswith('$') and v not in {'on','off','ON','IN','NEAR'}]
            entities=mentioned(text,all_values)
            burner='burner' in text and ('burner' in m['bindings'].get('$fixture',''))
            if not entities or mentioned(text,values) or burner:
                excluded.append({'mechanism_ref':m['id'],'api_call':m['api_call'],'prohibition':prohibition['text']});break
    ids={r['mechanism_ref'] for r in excluded}
    return [m for m in pool if m['id'] not in ids],excluded


def semantic_graph(pool,selected,scores):
    by_id={m['id']:m for m in pool};nodes=[];edges=[];unresolved=[]
    for id in selected:
        m=by_id[id];n={k:deepcopy(m[k]) for k in ['action','api_call','bindings','conditions','effects','role_types','role_type_constraints'] if k in m}
        n.update({'id':'semantic_'+id,'type':'action','mechanism_ref':id,'phase':None,'sources':['semantic'],
                  'score':scores.get(id,0),'candidate_only':True,'execution_verified':False})
        nodes.append(n)
    producers=defaultdict(list)
    for n in nodes:
        for fx in n['effects']:
            p=effect_predicate(fx)
            if p:producers[predicate_key(p)].append(n['id'])
    for n in nodes:
        for cond in n.get('conditions',[]):
            p=cond['predicate'];key=predicate_key(p)
            choices=[id for id in producers.get(key,[]) if id!=n['id']]
            resource=key and key[0]=='attribute' and key[1]=='agent' and key[2]=='gripper_empty'
            if len(choices)==1 and not resource:
                edges.append({'source':choices[0],'target':n['id'],'type':'candidate_precedes','predicate':p,
                              'basis':'declared_effect_condition','condition_basis':cond.get('basis'),
                              'depends_on_initial_state':True,'causal_necessity_verified':False})
            else:unresolved.append({'consumer':n['id'],'predicate':p,'candidate_producers':choices,
                                    'reason':'candidate graph is not a selected plan'})
    g={'nodes':nodes,'edges':edges,'goal_states':[],'route':'semantic','order':order_summary(nodes,edges),
       'unresolved_conditions':unresolved,'unresolved_goals':[],'support_status':'candidate_only',
       'complete_execution_plan':False,'dependency_scope':'candidate effect-condition matches; similarity is never an ordering edge'}
    g['mermaid']=mermaid(g);return g


def build_response(description,pool,scores,semantic_selected,initial_state=None,mode='all',max_nodes=32):
    fuzzy=semantic_graph(pool,semantic_selected,scores)
    pure=extract_goals(description,pool)
    integrated=extract_goals(description,pool,scores)
    pure_pool,pure_excluded=constrained_pool(pool,pure)
    hybrid_pool,hybrid_excluded=constrained_pool(pool,integrated)
    goal=DependencyBuilder(pure_pool,max_nodes=max_nodes).build(pure['goals'],initial_state, pure['temporal'],'goal')
    hybrid=DependencyBuilder(hybrid_pool,scores,max_nodes).build(integrated['goals'],initial_state,integrated['temporal'],'hybrid')
    goal['excluded_by_prohibitions']=pure_excluded;hybrid['excluded_by_prohibitions']=hybrid_excluded
    for g,extraction in [(goal,pure),(hybrid,integrated)]:
        g['final_goal_states']=extraction['final_goal_states'];g['goal_extraction']=extraction
        g['binding_equalities']=[note['binding_equality'] for note in extraction['ambiguities'] if 'binding_equality' in note]
        g['scene_binding_requirements']=[{'goal':goal['id'],'constraints':goal['binding_constraints']} for goal in extraction['goals'] if goal['binding_constraints']]
        for requirement in extraction['unmodeled_text_requirements']+extraction['prohibitions']:
            g['unresolved_goals'].append(requirement);g['support_status']='unresolved'
        if extraction['unparsed_description']:
            g['unresolved_goals'].append({'description':description,'reason':'no text-grounded goal predicate; semantic candidates are diagnostic alternatives'})
            g['support_status']='unresolved'
        fuzzy_calls={n['api_call'] for n in fuzzy['nodes']}
        for n in g['nodes']:
            n['sources']=['goal','semantic'] if n['api_call'] in fuzzy_calls else ['goal']
            n['selection_basis']='effect-compatible producer; semantic rank disambiguates' if g is hybrid else 'effect-compatible producer without semantic ranking'
        g['conditional_bindings']=[{'node':n['id'],'bindings':n['bindings'],'constraints':n.get('role_type_constraints',{})}
                                   for n in g['nodes'] if any(v.startswith('$') for v in n['bindings'].values())]
        if g['conditional_bindings'] and g['support_status']=='complete':g['support_status']='conditional'
    graphs={'semantic':fuzzy,'goal':goal,'hybrid':hybrid};main=graphs['hybrid' if mode=='all' else mode]
    hybrid_calls={n['api_call'] for n in hybrid['nodes']}
    return {'schema_version':'causground-task-dependency-interface-v1','input':description,'mode':mode,'graph':main,
            'graphs':graphs if mode=='all' else {mode:main},
            'semantic_candidates_not_required':[n for n in fuzzy['nodes'] if n['api_call'] not in hybrid_calls],
            'task_id_used':False,'episode_labels_used':False,'checker_used':False,
            'execution_verified':False,'ordering_policy':'declared state contracts and requested text order; unknown initial conditions remain conditional'}
