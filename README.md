# Non-Invasive Medical Digital Twins using Physics-Informed Self-Supervised Learning

**Overview**: A digital twin is a virtual replica of a real-world physical phenomena that uses mathematical modeling to characterize and simulate its defining features. By constructing digital twins for disease processes, we can perform in-silico simulations that mimic patients' health conditions and counterfactual outcomes under hypothetical interventions in a virtual setting. This eliminates the need for invasive procedures or uncertain treatment decisions. In this paper, we propose a method to identify digital twin model parameters using only noninvasive patient health data. We approach the digital twin modeling as a {\it composite inverse problem}, and observe that its structure resembles pretraining and finetuning in self-supervised learning (SSL). Leveraging this, we introduce a {\it physics-informed SSL} algorithm that initially pretrains a neural network on the pretext task of solving the physical model equations. Subsequently, the model is trained to reconstruct low-dimensional health measurements from noninvasive modalities while being constrained by the physical equations learned in pretraining. We apply our method to identify digital twins of cardiac hemodynamics using noninvasive echocardiogram videos, and demonstrate its utility in unsupervised disease detection and in-silico clinical trials.

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
The CAMUS data is available at [https://www.creatis.insa-lyon.fr/Challenge/camus/](https://www.creatis.insa-lyon.fr/Challenge/camus/)

The EchoNet data is available at [https://echonet.github.io/dynamic/](https://echonet.github.io/dynamic/)

## Model Training Directory

<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" rowspan="2">Interpolators</th>
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
    <td align="center" rowspan="1"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/interpolator_3param.py">View Code</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_3param_echonet.py">View Code</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_3param_camus.py">View Code</a></td>
  </tr>
  <tr>
    <td align="center">7-param</td>
    <td align="center" rowspan="1"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/interpolator_7param.py">View Code</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_7param_echonet.py">View Code</a></td>
    <td align="center"><a href="https://github.com/AlaaLab/Clinical-Sim2Real_exp/blob/master/training/pssl_7param_camus.py">View Code</a></td>
  </tr>
</tbody>
</table>




## Demo
A simulation of the pressure-volume loops with artificial volume oscillations can be found here: https://www.desmos.com/calculator/dgfbaot4zf
