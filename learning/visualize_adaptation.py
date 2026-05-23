from mujoco_playground import registry

import time, os
import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from controller.offline_controller import RobotController, OfflineRobotController
from controller.plots import PLOT_DATA_DIR

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


def interactive_visualization_sequence(env_sequence, controller, time_per_env=15.0):
    """
    Opens an interactive MuJoCo viewer and progresses through a sequence of environments.
    
    Args:
        env_sequence: List of tuples [(env_object, "EnvName"), ...]
        controller: The RobotController instance.
        time_per_env: How long (in simulation seconds) to spend in each environment.
    """
    # Ensure JIT functions exist for all environments in the sequence
    for env in env_sequence:
        if not hasattr(env, 'jit_reset'):
            env.jit_reset = jax.jit(env.reset)
            env.jit_step = jax.jit(env.step)

    # Setup Initial Environment
    env_idx = 0
    current_env = env_sequence[env_idx]
    controller.setEnv(current_env)

    # Initialize viewer with the first environment's model
    if hasattr(current_env, 'mj_model'):
        model = current_env.mj_model
    else:
        model = current_env.unwrapped.mj_model
        
    data = mujoco.MjData(model)
    rng = jax.random.PRNGKey(35)  # Fixed seed for reproducibility
    
    # Initialize the Simulation State
    rng, key1 = jax.random.split(rng)
    state = current_env.jit_reset(key1)
    
    env_timer = 0.0
    control_dt = getattr(current_env, 'dt', model.opt.timestep)
    
    print(f"Simulation running at control DT: {control_dt:.4f}s")
    print(f"Starting in {current_env.name}. Will hot-swap every {time_per_env} seconds.")

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] =[0, 0, 0.5]
        
        while viewer.is_running():
            # 1. Sequence Hot-Swap Logic
            if env_timer >= time_per_env:
                env_idx += 1
                
                # Check if we have completed the entire sequence
                if env_idx >= len(env_sequence):
                    print("\n--- EXPERIMENT SEQUENCE COMPLETE ---")
                    break
                    
                next_env = env_sequence[env_idx]
                print(f"\n--- HOT SWAPPING PHYSICS TO {next_env.name} ---")
                
                # Create a structurally valid state for the new environment
                rng, swap_rng = jax.random.split(rng)
                state2 = next_env.jit_reset(swap_rng)

                safe_qpos = state.data.qpos
                
                # 2. Teleport X and Y back to the center of the map
                safe_qpos = safe_qpos.at[0].set(0.0) # X
                safe_qpos = safe_qpos.at[1].set(0.0) # Y

                # Inject kinematics! Preserve the robot's physical momentum and posture.
                new_data = state2.data.replace(
                    qpos=safe_qpos,
                    qvel=state.data.qvel,
                    ctrl=state.data.ctrl,
                    time=state.data.time
                )
                state = state2.replace(
                    data=new_data,
                    obs=state.obs,
                    info=state.info
                )
                
                current_env = next_env
                controller.setEnv(current_env)
                env_timer = 0.0  # Reset timer for the next environment

            # 2. Control Loop
            state.info["command"] = controller.cmd
            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = controller.inference(state.obs, act_rng)[0]

            # Step environment
            prev_obs = state.obs
            state = current_env.jit_step(state, action)
            
            # Controller online monitoring and adaptation logic
            controller.control_loop(prev_obs, action, state)

            if getattr(controller, 'request_env_reset', False):
                # Clear the flag so it only resets once
                controller.request_env_reset = False

                if state.done:
                    print("\n>>> Fall after adaptation: Restarting robot to default standing pose...")
                    
                    # Get a fresh random key and reset the environment
                    rng, reset_rng = jax.random.split(rng)
                    state = current_env.jit_reset(reset_rng)
                    controller.detector.reset()

            # 3. Viewer Sync
            state.data.qpos.block_until_ready()
            
            # Extract qpos/qvel to CPU and copy to viewer
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)

            mujoco.mj_forward(model, data)
            viewer.sync()

            # 4. Timing
            env_timer += control_dt
            elapsed = time.time() - step_start
            
            # Sleep only the remaining time to match real-time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

def interactive_visualization(env, controller=None, resetNum=-1, jit_inference=None):
    """
    Opens an interactive MuJoCo viewer for a JAX-based environment.
    
    Args:
        env: The mujoco_playground environment (unwrapped or wrapped).
        params: The trained policy parameters.
        inference_fn: The function make_inference_fn(params, deterministic=True).
    """

    if controller is None:
        obs_shape, act_shape = env.observation_size, env.action_size
        controller = RobotController(obs_shape, act_shape, jit_inference=jit_inference)

    controller.setEnv(env)

    # Get the underlying standard MuJoCo model for the viewer
    if hasattr(env, 'mj_model'):
        model = env.mj_model
    else:
        # Fallback if environment is wrapped, try to access via unwrapped
        model = env.unwrapped.mj_model
        
    data = mujoco.MjData(model)
    rng = jax.random.PRNGKey(0)
    
    if not hasattr(env, 'jit_reset'):
        env.jit_reset = jax.jit(env.reset)
        env.jit_step = jax.jit(env.step)

    # Initialize the Simulation State
    rng, key1 = jax.random.split(rng)
    state = env.jit_reset(rng)
    
    reset_timer = 0
    control_dt = getattr(env, 'dt', model.opt.timestep)
    print(f"Simulation running at control DT: {control_dt:.4f}s")

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Initialize viewer camera if needed
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0, 0, 0.5]
        
        step_start = time.time()
        
        while viewer.is_running():
            # Reset after 10 seconds
            if reset_timer >= 10:
                #print("Resetting")
                rng, key1 = jax.random.split(rng)
                state = env.jit_reset(rng)
                reset_timer = 0
                resetNum -= 1

                if resetNum == 0:
                    break

            # Instruct robot to go always forwards
            state.info["command"] = controller.cmd

            step_start = time.time()
            rng, act_rng = jax.random.split(rng)
            
            # Run Policy
            action = controller.inference(state.obs, act_rng)[0]

            # Step environment
            prev_obs = state.obs
            state = env.jit_step(state, action)
            controller.control_loop(prev_obs, action, state)

            # Ensure computation is done before we try to read it
            state.data.qpos.block_until_ready()
            
            # Extract qpos/qvel from JAX to Numpy (CPU)
            # Copy to the viewer's data structure
            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)

            # Forward kinematics (compute world positions of geoms based on qpos)
            mujoco.mj_forward(model, data)

            # Sync the viewer
            viewer.sync()

            reset_timer += control_dt
            elapsed = time.time() - step_start
            # Sleep only the remaining time to match real-time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

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

    cmds = [jp.array([0.6, 0., 0.])]#, jp.array([.7, 0., 0.]), jp.array([.4, 0., 0.]), jp.array([.25, 0., 0.])]


    for cmd in cmds:
        controller = OfflineRobotController(obs_shape, act_shape, initial_pair="FlatTerrain", 
                                            generatePlots = False, cmd = cmd)
        
        # Define the chronological journey of the robot
        env_sequence =[
            flat_env,
            rough_env,
            slippery_env,
            slippery_env,
            rough_env,
            env_broken
        ]
        
        # Run the continuous sequence
        interactive_visualization_sequence(
            env_sequence=env_sequence, 
            controller=controller, 
            time_per_env=15.0 # 15 seconds gives plenty of time for PPO to trigger and stabilize
        )
        controller.export_history(os.path.join(PLOT_DATA_DIR, f"adaptSequence5.pkl"))

if __name__ == "__main__":
    main()