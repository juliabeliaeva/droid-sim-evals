"""Evaluation script that uses the tiptop websocket server for planning.

This script connects to a running tiptop websocket server, sends initial
observations (RGB, depth, camera params, task instruction), receives a
trajectory plan, and executes it in the Isaac Sim environment.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_eval.py --scene 1 --ws-host localhost --ws-port 8765
"""
from pathlib import Path

import argparse
import json
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
from src.sim_evals.sim_utils import settle_sim
from src.visual_utils import add_top_padding, overlay_timer_ms, overlay_prompt


def main(
        instructions: list[str],
        episodes: int = 1,
        headless: bool = True,
        scene: int = 1,
        variants: int = 1,
        ws_host: str = "localhost",
        ws_port: int = 8765,
):
    """Run evaluation using tiptop websocket server.

    Args:
        episodes: Number of episodes to run
        headless: If True (default), runs without the Isaac Sim GUI and only saves a video file.
            Set to False to open the Isaac Sim viewport for live visualization:
            ``uv run python tiptop_eval.py --headless False``
        scene: Scene number (1-5)
        variants: Number of variants to run
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
        instructions: A list of natural language task instructions.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Tiptop websocket evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaacsim.core.utils.stage import create_new_stage

    # Connect to tiptop websocket server
    logger.info(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
    client = TiptopWebsocketClient(host=ws_host, port=ws_port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")

    summary = []

    for variant in range(variants):
        if variant > 0:
            # env.close() releases the simulation context but leaves its USD stage open.
            create_new_stage()
        logger.info(f"Running scene {scene} variant {variant}")
        result = run_scene(args_cli.device, client, scene, variant, instructions, episodes, headless, video_dir)
        summary.extend(result)

    (video_dir / f"summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    client.close()
    simulation_app.close()


def run_scene(device: str, client: TiptopWebsocketClient, scene: int, variant: int, instructions: list[str],
              episodes: int, headless: bool, video_dir: Path) -> list[dict]:
    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("DROID", device=device, num_envs=1, use_fabric=True, )

    env_cfg.set_scene(str(scene), variant)
    env_cfg.episode_length_s = 180.0
    env_cfg.seed = 42
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials

    video_dir.mkdir(parents=True, exist_ok=True)

    max_steps = env.env.max_episode_length
    video_fps = 15

    summary = []
    with torch.no_grad():
        for i_idx, instruction in enumerate(instructions):
            for ep in range(episodes):
                obs, _ = env.reset()
                frame_idx = 0
                obs = settle_sim(env, obs, reset_episode_buf=True)
                plan_failed = False
                metadata = dict(scene=scene, variant=variant, episode=ep, prompt=instruction)
                video = []
                for i in tqdm(range(max_steps), desc=f"Episode {ep + 1}/{episodes}"):
                    try:
                        ret = client.infer(obs, instruction)
                    except Exception as e:
                        metadata["error"] = str(e)
                        logger.error(f"Planning failed for episode {ep + 1}: {e}. Skipping.")
                        plan_failed = True
                        break

                    if client.plan_done:
                        logger.info(f"Plan fully executed at step {frame_idx}")
                        break

                    viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                    viz = add_top_padding(viz, pad_px=40)
                    elapsed_ms = int(frame_idx * 1000 / video_fps)
                    overlay_timer_ms(viz, elapsed_ms)
                    overlay_prompt(viz, instruction)
                    if not headless:
                        try:
                            cv2.imshow("Camera View", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                            cv2.waitKey(1)
                        except cv2.error:
                            pass

                    video.append(viz)
                    frame_idx += 1

                    action = torch.tensor(ret["action"])[None]
                    obs, _, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break

                client.reset()
                metadata["plan_success"] = not plan_failed
                episode_name = f"s{scene}_v{variant}_p{i_idx}_ep{ep}"
                if plan_failed:
                    image_file = video_dir / f"tiptop_scene_{episode_name}_failure.png"
                    frame = obs["policy"]["external_cam"][0].detach().cpu().numpy()
                    mediapy.write_image(image_file, frame)
                    metadata["image_file"] = str(image_file)
                else:
                    video_file = video_dir / f"tiptop_scene_{episode_name}.mp4"
                    mediapy.write_video(video_file, video, fps=video_fps)
                    metadata["video_file"] = str(video_file)
                    logger.info(f"Saved video to {video_file}")

                (video_dir / f"metadata_{episode_name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
                summary.append(metadata)
    env.close()

    return summary


if __name__ == "__main__":
    tyro.cli(main)
