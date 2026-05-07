import jax, random
import jax.numpy as jp
from controller.robot_controller import RobotController
from visualize_adaptation import load_env, IMPL

def run_trial(env1, env2, start_env_key, target_env_key, velocity, use_adaptation, seed, controller):
    """
    Runs a single headless trial. 
    Phase 1: 250 steps in env1 to reach steady state.
    Phase 2: 500 steps in env2 to test adaptation.
    """
    controller.setEnv(env1)
    controller.set_policy(start_env_key)
    controller.detector.reset()

    controller.errors = {name: [] for name in controller.pol_names}
    controller.sampling = False
    controller.cmd = jp.array(velocity)
    controller.deploy = use_adaptation
    
    controller.smooth_errors = {name: [] for name in controller.pol_names}
    controller.env_changes = []
    controller.drift_indices = []
    controller.contact_history = []
    controller.policy_history = []
    controller.gp_states = []
    
    rng = jax.random.PRNGKey(seed)
    rng, key_reset = jax.random.split(rng)
    
    # Initialize State
    state = env1.jit_reset(key_reset)
    fell = False

    # Phase 1: Walk in starting environment
    for _ in range(250):
        rng, act_rng = jax.random.split(rng)
        state.info["command"] = controller.cmd
        action = controller.inference(state.obs, act_rng)[0]
        prev_obs = state.obs
        state = env1.jit_step(state, action)
        
        if use_adaptation:
            controller.control_loop(prev_obs, action, state)
            
        if state.done:
            fell = True
            break
            
    # Phase 2: Domain Shift!
    if not fell:

        if use_adaptation:
            controller.setEnv(env2)

        # 1. Reset env2 to get a structurally valid JAX state tree for the new model
        rng, swap_rng = jax.random.split(rng)
        state2 = env2.jit_reset(swap_rng)
        
        # 2. Transfer the exact physical kinematics (posture, velocity, control, time)
        # qpos (19) and qvel (18) are identical across all Go2 environments!
        new_data = state2.data.replace(
            qpos=state.data.qpos,
            qvel=state.data.qvel,
            ctrl=state.data.ctrl,
            time=state.data.time
        )
        
        # 3. Overwrite the state, carrying over the observations and info history 
        state = state2.replace(
            data=new_data,
            obs=state.obs,
            info=state.info
        )
            
        for _ in range(600):
            rng, act_rng = jax.random.split(rng)
            state.info["command"] = controller.cmd
            action = controller.inference(state.obs, act_rng)[0]
            prev_obs = state.obs
            
            # Hot-swap the environment by stepping env2 with the current state!
            state = env2.jit_step(state, action)
            
            if use_adaptation:
                controller.control_loop(prev_obs, action, state)
                
            if state.done:
                fell = True
                break

    # Determine if adaptation was semantically correct
    correct_adapt = False
    if use_adaptation:
        final_policy = controller.active_wm[0]
        # Check if the target environment name is in the final policy's name
        if target_env_key in final_policy.split('_AdaptedFrom_')[0]:
            correct_adapt = True

    return fell, correct_adapt

def main():
    print("Pre-compiling environments...")
    envs = {
        "FlatTerrain": load_env("Go2StrollFlatTerrain", IMPL),
        "RoughTerrain": load_env("Go2StrollRoughTerrain", IMPL),
        "SlipperyTerrain": load_env("Go2StrollSlipperyTerrain", IMPL),
        "BlockedKnee": load_env("Go2StrollFlatTerrain", IMPL, breakLeg=True)
    }

    obs_shape, act_shape = envs["FlatTerrain"].observation_size, envs["FlatTerrain"].action_size
    
    # Initialize controller. Set deploy=use_adaptation to toggle the system.
    controller = RobotController(
        obs_shape, act_shape, 
        initial_pair="FlatTerrain", 
        generatePlots=False, 
        cmd=None
    )

    transitions = [
        ("FlatTerrain", "SlipperyTerrain"),
        ("FlatTerrain", "BlockedKnee"),
        ("FlatTerrain", "RoughTerrain"),
    ]
    
    velocities = [
        ([0.3, 0.0, 0.0], "Low (0.3 m/s)"),
        ([0.6, 0.0, 0.0], "Medium (0.6 m/s)"),
        ([1.0, 0.0, 0.0], "High (1.0 m/s)")
    ]

    # The parameters are: [increment_samples, discarded_samples]
    configs = [
        [15, 0],
        [20, 5],
        [20, 0]
    ]

    num_trials = 50 # Start with 50 to test. Increase to 1000 for final thesis data!

    print(f"\n--- Starting Headless Evaluation ({num_trials} trials per config) ---")
    
    for start_env_key, target_env_key in transitions:
        print(f"\nTransition: {start_env_key} -> {target_env_key}")
        print("-" * 60)
        
        env1 = envs[start_env_key]
        env2 = envs[target_env_key]
        
        for vel_array, vel_name in velocities:
            
            # Baseline: NO ADAPTATION
            baseline_falls = 0
            for i in range(num_trials):
                fell, _ = run_trial(env1, env2, start_env_key, target_env_key, vel_array, use_adaptation=False, 
                                    seed=i, controller=controller)
                if fell: baseline_falls += 1
            

            for increment_samples, discarded_samples in configs:
                controller.increment_samples = increment_samples
                controller.noisy_samples = discarded_samples
                
                # Test: WITH ADAPTATION
                adapt_falls = 0
                adapt_correct = 0
                for i in range(num_trials):
                    fell, correct = run_trial(env1, env2, start_env_key, target_env_key, vel_array, use_adaptation=True, 
                                            seed=i+1000, controller=controller)
                    if fell: adapt_falls += 1
                    if correct: adapt_correct += 1
                    
                b_fall_pct = (baseline_falls / num_trials) * 100
                a_fall_pct = (adapt_falls / num_trials) * 100
                a_succ_pct = (adapt_correct / num_trials) * 100
                
                print(f"[{vel_name}]")
                print(f"Increment samples: {controller.increment_samples}, noisy samples = {controller.noisy_samples}")
                print(f"  Baseline Falls:   {b_fall_pct:>5.1f}%")
                print(f"  Falls With Adaptation:     {a_fall_pct:>5.1f}%")
                print(f"  Correct Policy: {a_succ_pct:>5.1f}%")

if __name__ == "__main__":
    main()