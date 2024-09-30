# Med-Real2Sim: Non-Invasive Medical Digital Twins using Physics-Informed Self-Supervised Learning

**Overview**: A digital twin is a virtual replica of a real-world physical phenomena that uses mathematical modeling to characterize and simulate its defining features. By constructing digital twins for disease processes, we can perform in-silico simulations that mimic patients' health conditions and counterfactual outcomes under hypothetical interventions in a virtual setting. This eliminates the need for invasive procedures or uncertain treatment decisions. In this paper, we propose a method to identify digital twin model parameters using only noninvasive patient health data. We approach the digital twin modeling as a composite inverse problem, and observe that its structure resembles pretraining and finetuning in self-supervised learning (SSL). Leveraging this, we introduce a physics-informed SSL algorithm that initially pretrains a neural network on the pretext task of solving the physical model equations. Subsequently, the model is trained to reconstruct low-dimensional health measurements from noninvasive modalities while being constrained by the physical equations learned in pretraining. We apply our method to identify digital twins of cardiac hemodynamics using noninvasive echocardiogram videos, and demonstrate its utility in unsupervised disease detection and in-silico clinical trials.

## Set up the environment

To ensure compatibility, you can install specific versions of the dependencies by running the following commands:

```shell
pip install matplotlib==3.7.1
pip install torch==2.1.0+cu121
pip install scikit-image==0.19.3
pip install numpy==1.25.2
pip install scipy==1.11.4
pip install pandas==1.5.3
pip install torchvision==0.16.0+cu121
```

## Data
The preprocessed CAMUS data in two-chamber and four-chamber views can be downloaded at [2CH](https://drive.google.com/drive/folders/1bN9qUvPrajReSZPDskFT8N8cag6mpRaJ?usp=drive_link) and [4CH](https://drive.google.com/drive/folders/14NMaOJ-NfwO5qHPcOTarCMBLvJWJ1Rs0?usp=drive_link)

The EchoNet data is available at [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)

## Model Training and Inference Instructions

### Physics-Informed Pretext Task
```shell
python training/physics_pretext_7param.py --output_path /path/to/pretext_model
python training/physics_pretext_3param.py --output_path /path/to/pretext_model
```
### Physics-Guided Fine-Tuning with EchoNet or CAMUS dataset
**Step 1.** Download datasets and follow the instructions from the links provided in the **Data Section** and save to /path/to/echonet_input and /path/to/camus_input


**Step 2.** Run training scripts for both datasets
```shell
python training/pssl_7param_echonet.py --output_path /path/to/output --batch_size 100 --pretext_model_path /path/to/pretext_model --num_epochs 200 --learning_rate 0.001 --ID full_echonet_7param_Vloss --echonet_input_directory /path/to/echonet_input
python training/pssl_7param_camus.py --output_path /path/to/output --batch_size 100 --pretext_model_path /path/to/pretext_model --num_epochs 200 --learning_rate 0.001 --ID full_camus_7param_Vloss --echonet_input_directory /path/to/camus_input

```

### Use cases
**Step 1.** Create parameter datasets 
```shell
python eval/create_datasets/weight_to_param_echonet.py --output_path /path/to/output --ID create_dataset_echonet_7param --echonet_input_directory /path/to/echonet_input --model_path /path/to/model --pretext_model_path /path/to/pretext_model
python eval/create_datasets/weight_to_param_camus.py --output_path /path/to/output --ID create_dataset_echonet_7param --echonet_input_directory /path/to/camus_input --model_path /path/to/model --pretext_model_path /path/to/pretext_model
```
**Step 2** Visualize PV loops for patients

Some pretrained weights can be found here:
<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" rowspan="2">pretext models</th>
    <th align="center" style="text-align:center" colspan="2">P-SSL 3DCNNs</th>
  </tr>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center">Echonet</th>
    <th align="center" style="text-align:center">CAMUS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">3-param</td>
    <td align="center" rowspan="1"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pretext_model_3param.py">Download</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_3param_echonet.py">Download</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_3param_camus.py">Download</a></td>
  </tr>
  <tr>
    <td align="center">7-param</td>
    <td align="center" rowspan="1"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pretext_model_7param.py">Download</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_7param_echonet.py">Download</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_7param_camus.py">Download</a></td>
  </tr>
</tbody>
</table>




## Demo
A simulation of model inference and of the pressure-volume loops with artificial volume oscillations can be found here: https://huggingface.co/spaces/alaa-lab/Med-Real2Sim # https://www.desmos.com/calculator/dgfbaot4zf
