# OMoE-Net: Orthogonally Disentangled Mixture-of-Experts Network for Image Restoration under Low-Level Degradations



<hr />

> **Abstract:** *Unified image restoration under low-level degradations aims to recover images corrupted by noise, blur, and compression within a single model. However, existing unified restoration methods may suffer from performance saturation or degradation as more degradation types are introduced, due to optimization conflicts and representation interference among different restoration objectives. To address this issue, we propose OMoE-Net, an orthogonally disentangled mixture-of-experts network for unified restoration. It incorporates an orthogonally disentangled expert module to encourage diverse and non-overlapping representations, reducing interference among restoration tasks. A shared expert branch is introduced to capture common restoration priors, while a degradation-aware path controller dynamically selects suitable expert combinations for each input. This design promotes effective coordination between shared knowledge and task-specific processing, improving scalability and robustness in unified restoration. Experiments on CBSD68, BSD68, Urban100, and Kodak24 demonstrate that OMoE-Net achieves superior and more stable performance than representative image restoration methods.* 
<hr />

## Network Architecture
<img src="https://github.com/taol-bee/OMoE-Net/figs/OMoE-Net.svg">

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the de_type argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (gsn, sp, jpeg, gb, mb).

Example Usage: If we only want to train on gsn and gb:
```
python train.py --de_type gsn gb
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. To perform the evaluation, use
```
python test.py --ckpt xx/xxxxx.ckpt --de_types [n] --offline_dir data/test_[m]
```
``--ckpt``: Path to the trained model checkpoint file.

``--de_types [n]``: Specify the degradation task(s) for testing.

``n`` can be one or multiple tasks from the 5 supported degradations: gsn, sp, jpeg, gb, mb.

You can test single task or multiple tasks jointly.

``--offline_dir data/test_[m]``: Specify the test dataset directory.

``m`` is the name of the test set: bsd68, urban100, kodak24, cbsd68.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --ckpt OMoE-Net/gsn-last.ckpt --de_types gsn --offline_dir data/test_bsd68
```



## Results
<details>
<summary><strong>Qualitative comparison resultss</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/results3.svg"> 

</details>
<details>
<summary><strong>Visualization diagram of different degradation characteristics</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/results2.png"> 

</details>

<details>
<summary><strong>Table1</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table1.png"> 

</details>

<details>
<summary><strong>Table2</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table2.png"> 

</details>

<details>
<summary><strong>Table3</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table3.png">

</details>

<details>
<summary><strong>Table4</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table4.png">

</details>

<details>
<summary><strong>Table5</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table5.png"> 

</details>

<details>
<summary><strong>Table6</strong> (click to expand) </summary>
<img src="https://github.com/taol-bee/OMoE-Net/figs/table6.png"> 

</details>



