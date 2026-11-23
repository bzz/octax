"""Check that recorded pixels/actions/states reproduce the corrected emulator."""
import argparse
import json
from pathlib import Path

from human_demonstrations import validate_demonstration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='One episode NPZ or a session directory.')
    args = parser.parse_args()
    files = [args.path] if args.path.is_file() else sorted(args.path.glob('episode_*.npz'))
    if not files:
        parser.error('No recorded episodes found.')
    for path in files:
        print(json.dumps(validate_demonstration(path)), flush=True)
