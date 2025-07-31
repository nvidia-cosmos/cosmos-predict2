# Video2World Post-training for Action-conditioned Video Prediction

We provide an example post-training instruction from a pre-trained video2world checkpoint.

## 1. Preparing Data
### 1.1 Download Bridge training dataset
We use the train/validation splits of the Bridge dataset from IRASim for action-conditioned post-training.
To download and prepare the dataset, run the following commands under the `cosmos-predict2/` directory:
```
wget https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/opensource_IRASim_v1/bridge_train_data.tar.gz
mv bridge_train_data.tar.gz datasets/
cd datasets
tar -xvzf bridge_train_data.tar.gz -C .
mv opensource_robotdata/bridge ./
```

Your dataset directory structure should look like this:
```
datasets/bridge/
├── annotations/
│   ├── *.json
├── videos/
    ├── *.mp4
```

Each JSON file in the `annotations/` folder contains the end-effector pose and gripper width of the robot arm for each frame in the corresponding video.
Specifically, each file includes:
- `state`: The end-effector pose of the robot arm at each timestep, represented as [x, y, z, roll, pitch, yaw].
    - (x, y, z) denotes the gripper's position in world coordinates.
    - (roll, pitch, yaw) describes its orientation in Euler angles.
- `continuous_gripper_state`: The width of the gripper at each timestep, indicating whether it is open or closed. A value of 0 means the gripper is open, and 1 means it is closed.
- `action`: The gripper's displacement at each timestep.
    - The first six dimensions represent displacement in (x, y, z, roll, pitch, yaw) within the gripper coordinate frame.
    - The last (seventh) dimension is a binary value indicating whether the gripper should open (1) or close (0).

We use this information as conditioning input for video generation.


## 2. Post-training

##### Cosmos-Predict2-2B-Video2World
Run the following command to launch an example post-training job using the Bridge dataset:
```bash
torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment="predict2_video2world_2b_action_conditioned_training"
```
See `cosmos_predict2/configs/action_conditioned/defaults/data.py` to understand how the dataloader is defined.
To add action as additional condition, we create new `conditioner` to support action in `cosmos_predict2/configs/action_conditioned/defaults/conditioner.py`.

##### Checkpoint Output Structure
Checkpoints are saved to the following path:
```
checkpoints/PROJECT/GROUP/NAME
```
For the example command above:
- PROJECT: `posttraining`
- GROUP: `video2world`
- NAME: `predict2_video2world_2b_action_conditioned_training_${now:%Y-%m-%d}_${now:%H-%M-%S}`

##### Configuration Snippet
Below is a configuration snippet defining the experiment setup:
```python
predict2_video2world_2b_action_conditioned_training = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_action_conditioned_fsdp"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "bridge_train"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
        )
    ),
    job=dict(group="debug", name="predict2_video2world_2b_action_conditioned_training_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
)
```


## 3. Inference for Bridge
##### Cosmos-Predict2-2B-Video2World
To run inference using a post-trained checkpoint (e.g., at 1000 iterations), use the command below.
Specify the path to the checkpoint using the `--dit_path` argument:
```
python examples/video2world_action.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/predict2_video2world_2b_action_conditioned_training_${now:%Y-%m-%d}_${now:%H-%M-%S}/checkpoints/model/iter_000001000.pt" \
  --input_video datasets/bridge/videos/test/13/rgb.mp4 \
  --input_annotation datasets/bridge/annotation/test/13.json \
  --num_conditional_frames 1 \
  --save_path output/generated_video.mp4 \
  --guidance 0 \
  --seed 0 \
  --disable_guardrail \
  --disable_prompt_refiner
```
