import dataclasses
import logging
import os
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Serve an ActiveScale checkpoint over WebSocket."""

    config: str
    checkpoint: str
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    denoise_steps: int = 10


def create_policy(args: Args) -> _policy.Policy:
    if args.denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive")
    return _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.denoise_steps},
    )


def create_server_metadata(args: Args, policy: _policy.Policy) -> dict:
    """Include the actually loaded checkpoint in websocket metadata."""
    metadata = dict(policy.metadata or {})
    metadata["action_sampling"] = {"num_steps": args.denoise_steps}
    checkpoint_dir = os.path.expanduser(args.checkpoint)
    if "://" not in checkpoint_dir:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
    metadata["deployment_checkpoint"] = {
        "policy_config": args.config,
        "checkpoint_dir": checkpoint_dir,
    }
    return metadata


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = create_server_metadata(args, policy)
    logging.info("Action flow-matching denoise steps: %d", args.denoise_steps)

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
