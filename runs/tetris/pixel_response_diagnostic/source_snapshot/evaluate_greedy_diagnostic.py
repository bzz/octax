"""Check deterministic action selection without updating any policy weights."""
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from evaluate_initial_policy import initial_state
from experiment_utils import evaluate, make_evaluator

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/tetris/v2_long_trace_512k_seed1'
CONTROL=ROOT/'runs/tetris/v2_credit_horizon_512k_seed1'
OUT=ROOT/'runs/tetris/pixel_response_diagnostic'


def greedy_action(agent,obs):
    return jnp.argmax(agent.actor._action_dist(agent.encode(obs)).logits,axis=-1)


class GreedyAgent:
    """Adapt only inference action choice to the existing raw-env evaluator."""
    def __init__(self,agent):
        self.agent=agent

    def apply(self,params,obs,rng):
        # make_evaluator consumes only tuple item zero. Keep its environment,
        # key splitting, observation conversion and termination logic exactly.
        return (self.agent.apply(params,obs,method=greedy_action),)


def main():
    cfg=OmegaConf.to_container(OmegaConf.load(RUN/'config.yaml'))
    raw,agent,initial=initial_state(cfg)
    checkpoints={k:(p/'checkpoint.msgpack').read_bytes() for k,p in [('long_trace',RUN),('credit_control',CONTROL)]}
    weights={'untrained':initial.agent_ts.params}
    weights.update({k:serialization.msgpack_restore(data)['train_state']['agent_ts']['params'] for k,data in checkpoints.items()})
    # Reproduce the latest stochastic eight-game monitoring result before
    # swapping the inference adapter; no separate environment implementation.
    recorded=json.loads((RUN/'evaluation.json').read_text())[-1]
    actual=evaluate(make_evaluator(raw,agent,8,10000),weights['long_trace'],'policy')
    assert actual['episodes']==recorded['episodes']
    # Argmax path must equal the most likely action from actual PPO log-probs.
    batch=initial.last_obs
    lp=jnp.stack([agent.apply(weights['long_trace'],batch,None,jnp.full((16,),i))[1] for i in range(5)],axis=-1)
    chosen=agent.apply(weights['long_trace'],batch,method=greedy_action)
    assert bool(jnp.all(chosen==jnp.argmax(lp,axis=-1)))
    evaluator=make_evaluator(raw,GreedyAgent(agent),128,50000)
    result=dict(seed_start=50000,episodes_per_policy=128,action_selection='argmax; ties select lowest index',
        checkpoint_sha256={k:hashlib.sha256(v).hexdigest() for k,v in checkpoints.items()},
        stochastic_monitoring_reproduced_exactly=True,argmax_matches_actual_policy_log_probabilities=True,
        policies={},limitations=['Same reused exploratory bank; not independent confirmation.',
                                'Evaluation action choice changes, not the trained objective or weights.',
                                'An improvement would require matched untrained-greedy and random controls, fresh games and independent train seeds.'])
    OUT.mkdir(parents=True,exist_ok=True)
    for name,params in weights.items():
        result['policies'][name]=evaluate(evaluator,params,'policy')
        (OUT/'greedy.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,result['policies'][name]['summary'],flush=True)
    for name,path in [('long_trace',RUN),('credit_control',CONTROL)]:
        assert (path/'checkpoint.msgpack').read_bytes()==checkpoints[name]


if __name__=='__main__':
    main()
