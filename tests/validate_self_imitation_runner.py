from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
import jax
import numpy as np
from flax import serialization
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from PIL import Image
import train
import train_self_imitation

out=args.out.resolve()

def tiny_gif(env,path,*args,**kwargs):
    # Rendering is not under test here; avoid replaying two extra illustrative games.
    Image.new('RGB',(1,1)).save(path)

with initialize_config_dir(version_base=None,config_dir=str(root/'conf')):
    baseline=OmegaConf.to_container(compose(config_name='credit_horizon'),resolve=True)
    replay=OmegaConf.to_container(compose(config_name='self_imitation'),resolve=True)
for module,cfg,name in [(train,baseline,'baseline'),(train_self_imitation,replay,'replay')]:
    cfg.pop('num_seeds')
    cfg.update(seed=1,total_timesteps=2048,eval_freq=2048,eval_episodes=1,run_dir=str(out/name))
    module.save_gif=tiny_gif
    module.run(cfg)
a=serialization.msgpack_restore((out/'baseline/checkpoint.msgpack').read_bytes())['train_state']
b=serialization.msgpack_restore((out/'replay/checkpoint.msgpack').read_bytes())['train_state']
assert jax.tree.structure(a)==jax.tree.structure(b)
for before,after in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
    np.testing.assert_array_equal(before,after)
assert train_self_imitation.verify_checkpoint_pair(out/'replay/checkpoint.msgpack')==2048
print('PASS: complete model/optimizer/RNG/emulator state matches baseline exactly before replay activation.')
print('PASS: paired model/replay checkpoint integrity and exact roundtrip checks.')
