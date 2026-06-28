from mujoco_playground import registry

import time, os
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from controller.offline_controller import RobotController, OfflineRobotController
from controller.plots import PLOT_DATA_DIR
import mediapy as media  # DeepMind's standard video writer

IMPL = "jax"
    
def load_env(env_name, impl, breakLeg=False):
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl
    if breakLeg:
        env_cfg["broken_leg"] = True

    env = registry.load(env_name, config=env_cfg)

    env.jit_reset = jax.jit(env.reset)
    env.jit_step = jax.jit(env.step)
    env.name = env_name if not breakLeg else "BlockedKnee_" + env_name

    return env


def record_visualization_sequence(env_sequence, controller, time_per_env=15.0, filename="adaptation_sequence.mp4"):
    """
    Runs the sequence completely offline and renders a stutter-free 50FPS MP4 video.
    """
    print(f"\nRecording video to '{filename}'.")
    
    for env in env_sequence:
        if not hasattr(env, 'jit_reset'):
            env.jit_reset = jax.jit(env.reset)
            env.jit_step = jax.jit(env.step)

    env_idx = 0
    current_env = env_sequence[env_idx]
    controller.setEnv(current_env)

    if hasattr(current_env, 'mj_model'):
        model = current_env.mj_model
    else:
        model = current_env.unwrapped.mj_model
        
    rng = jax.random.PRNGKey(33)
    rng, key1 = jax.random.split(rng)
    state = current_env.jit_reset(key1)
    
    control_dt = getattr(current_env, 'dt', model.opt.timestep)
    fps = int(1.0 / control_dt) # For 0.02 dt, this is 50 FPS
    
    # Initialize the writer (640x480 resolution)
    with media.VideoWriter(filename, shape=(480, 640), fps=fps) as writer:
        
        while env_idx < len(env_sequence):
            current_env = env_sequence[env_idx]
            print(f"\n--- RENDERING ENVIRONMENT: {current_env.name} ---")
            
            if hasattr(current_env, 'mj_model'):
                model = current_env.mj_model
            else:
                model = current_env.unwrapped.mj_model
                
            data = mujoco.MjData(model)
            
            # 2. Setup the Off-Screen Renderer
            renderer = mujoco.Renderer(model, height=480, width=640)
            
            # 3. Setup the Chase Camera
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance = 3
            cam.azimuth = 180.0
            cam.elevation = -10.0
            
            state.data.qpos.block_until_ready()
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(model, data)
            
            env_timer = 0.0

            # Draw video frames
            while env_timer < time_per_env:
                
                # Control Loop
                state.info["command"] = controller.cmd
                rng, act_rng = jax.random.split(rng)
                
                action = controller.inference(state.obs, act_rng)[0]
                prev_obs = state.obs
                state = current_env.jit_step(state, action)
                
                controller.control_loop(prev_obs, action, state)

                # Resurrection logic
                if getattr(controller, 'request_env_reset', False):
                    controller.request_env_reset = False
                    if state.done:
                        print(">>> Fall after adaptation: Restarting robot...")
                        rng, reset_rng = jax.random.split(rng)
                        state = current_env.jit_reset(reset_rng)
                        controller.detector.reset()

                # 4. Sync physics to CPU and render the frame
                state.data.qpos.block_until_ready()
                data.qpos[:] = np.array(state.data.qpos)
                data.qvel[:] = np.array(state.data.qvel)
                mujoco.mj_forward(model, data)

                # Make camera track the robot
                cam.lookat[0] = data.qpos[0]
                cam.lookat[1] = data.qpos[1]
                cam.lookat[2] = 0.5
                
                # Take the picture and append to the MP4!
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render()
                writer.add_image(pixels)

                env_timer += control_dt

            # Swapping environment physics
            renderer.close() # Free GPU memory for the old environment renderer
            env_idx += 1
            
            if env_idx < len(env_sequence):
                next_env = env_sequence[env_idx]
                rng, swap_rng = jax.random.split(rng)
                state2 = next_env.jit_reset(swap_rng)

                safe_qpos = state.data.qpos
                safe_qpos = safe_qpos.at[0].set(0.0) # X Teleport
                safe_qpos = safe_qpos.at[1].set(0.0) # Y Teleport

                new_data = state2.data.replace(
                    qpos=safe_qpos, qvel=state.data.qvel,
                    ctrl=state.data.ctrl, time=state.data.time
                )
                state = state2.replace(
                    data=new_data, obs=state.obs, info=state.info
                )
                controller.setEnv(next_env)

    print(f"\nEXPERIMENT COMPLETE! Video saved to: {filename}")

def main():
    '''
    Loads the omni-directional policy network and environment for the Go2 Stroll environment.
    Then, it simulates it in the Mujoco interactive viewer using JAX.
    '''

    env_name = "Go2StrollFlatTerrain"

    flat_env = load_env(env_name, IMPL)
    rough_env = load_env("Go2StrollRoughTerrain", IMPL)
    slippery_env = load_env("Go2StrollSlipperyTerrain", IMPL)
    env_broken = load_env(env_name, IMPL, breakLeg=True)

    obs_shape, act_shape = flat_env.observation_size, flat_env.action_size

    cmd = jp.array([.6, 0., 0.])


    controller = OfflineRobotController(obs_shape, act_shape, initial_pair="FlatTerrain", 
                                        generatePlots = False, cmd = cmd)
    
    # Define the chronological journey of the robot
    env_sequence =[
        flat_env,
        slippery_env
    ]
    
    # Run the continuous sequence
    record_visualization_sequence(
        env_sequence=env_sequence, 
        controller=controller, 
        filename="continual.mp4",
        time_per_env=6.0 # 15 seconds gives plenty of time for PPO to trigger and stabilize
    )
    controller.export_history(os.path.join(PLOT_DATA_DIR, f"continualSlides.pkl"))

if __name__ == "__main__":
    main()