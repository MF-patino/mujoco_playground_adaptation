# Online policy adaptation with Mujoco Playground 

This fork of Mujoco Playground provides a general framework for online adaptation of Go2 quadruped gaits in environments with changing terrain geometries and friction coefficients. The proposed solution bypasses catastrophic forgetting completely by building a scaffold that forever catalogs policies and also efficiently retrieves the best one (among hundreds or thousands) for a given environment.

## Instalation instructions

* Clone this repository and install it through the official instructions.
* Create a Python virtual environment: source ~/.venv/bin/activate
    ```sh 
    source ~/.venv/bin/activate
    ```
* Install the river online learning library: 
    - Install Rust: 
        ```sh 
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        ```
    - Choose option 1
    - Close the terminal and open a new one, then execute: 
        ```sh 
        uv pip install river
        ```

* Install scikit-learn and umap
    ```sh 
    uv pip install scikit-learn umap-learn
    ```

* Verify everything runs:
    ```sh 
    python learning/visualize_adaptation.py
    ```

* (Workaround) If there is a MuJoCo version mismatch, this may fix it:
    ```sh 
    uv pip install mujoc==3.7.0 mujoco-mjx==3.7.0
    ```

* (Optional) Install rscope:
    ```sh 
    uv pip install rscope
    ```

## Folders

### Policy-WM pairs (Catalog)

The online adaptation process uses pairs of policies and their respective world models to track their performance, and these pairs can be found in the `model_pairs` directory. This repo ships with it the policies for the Simple Catalog discussed in the thesis report.

If the online adaptation system finds an environment it does not recognize, it then automatically trains and generates a new policy-WM pair for it in this folder. These adapted pairs contain the keyword "AdaptedFrom" in their folder names.

### Plot data for visualization

The folder `plotData` holds all data files that were used to build the gait, GP search, policy embedding and drift detection plots in the thesis report paper. As for the subfolder `plotData/training`, it holds the results (mean episode rewards at each time step) from training each policy from scratch and with transfer learning 20 independent times. The contents of this folder are used to compute the statistical tests that prove transfer learning is more sample efficient that training from scratch.

### Code developed for the project

The folder `learning` holds many new scripts that are not shipped with the official MuJoCo Playground repo. Their usage is discussed in the following section.

As for the subfolders, `learning/controller` holds the code for adaptation to new domains and concept drift detection (`learning/controller/ks_detector`), while `learning/worldModel` contains the code for World Model training (`learning/worldModel/train_world_model.py`) and some configurations and folder paths that all modules have in common (`learning/worldModel/common.py`), as well as the code to extract transitions to build a dataset with which to train WMs (`learning/worldModel/rollout_saver.py`).

Finally, `mujoco_playground/_src/locomotion/go2/stroll.py` contains the reward function and code to train the movement policies for the Go2 robot. The environments used in this work were defined in the folder `mujoco_playground/_src/locomotion/go2/xmls`.

## Usage

### For deployment and visualization

* For deploying the online adaptation stack in simulation with continual learning:
    ```sh 
    python learning/visualize_adaptation.py
    ```

    This will create a new file in the `plotData` folder, which holds all collected information that can then be visualized.

* Visualizing plots (requires manually setting the file to visualize in the `PLOT_FILE` global variable): 
    ```sh 
    python learning/plot_graphs.py
    ```

### Generating the catalogs

All three catalogs evaluated in the thesis may be generated using the script launched as follows:
```sh 
python learning/generateCatalog.py
```

However, first one must set the global variable `ALL_FROM_SCRATCH` to `True` to train the catalog trained from scratch, or alternatively to `False` in order to train the entire Redundant Catalog. As the Simple Catalog is a subset of the Redundant Catalog, one can simply remove the resulting redundant policies from the `model_pairs` directory after training, or stopping the training manually when the first four policies are generated (which are the Simple Catalog).

### Evaluation and statistical tests

Once a catalog has been built, it is possible to compute the quantitative evaluation table for that catalog as follows: 
```sh 
python learning/evaluate_headless.py
```

It is also possible to run a statistical test to determine if the Simple Catalog is more sample efficient than the catalog trained from scratch. In order to do this, the `plotData/training` folder must be emptied and the `generateCatalog.py` has to be invoked with the global variable `DO_TRAINING_TRIALS` set to `True`. This will populate the folder, after which we can launch the following script which reads the contents of the folder to make the plot:
```sh 
python learning/stat_test.py