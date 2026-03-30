# Orthogonally Disentangled Mixture-of-Experts for Unified Image Restoration under Heterogeneous Degradations



<hr />

> **Abstract:** *Unified image restoration aims to recover high-quality images from multiple degradations within a shared model, which is important for practical deployment in complex real-world scenarios. However, existing methods often show unstable gains or even performance degradation as more heterogeneous degradation types are included in training, due to conflicting optimization demands in a shared parameter space. In this paper, we propose OMoE-Net, an orthogonally disentangled mixture-of-experts framework for unified image restoration under heterogeneous degradations. A shared experts branch and a degradation-specific expert branch are jointly introduced to model common restoration priors and degradation-related variations in a unified encoder-decoder architecture. Orthogonal regularization is applied to expert parameters to promote clearer expert specialization, and a degradation-aware path controller further selects Top-K experts while imposing orthogonal guidance on routed features to reduce representation overlap. Extensive experiments on multiple representative degradations demonstrate that OMoE-Net delivers better overall restoration performance and stronger robustness to increasing task diversity.* 
<hr />

## Network Architecture
<img src = "figs/OMoE-Net.svg"> 

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



## Qualitative comparison results
<details>
<img src = "figs/results3.svg"> 

</details>


## Visualization diagram of different degradation characteristics
<details>
<img src = "figs/result2.svg"> 

</details>






