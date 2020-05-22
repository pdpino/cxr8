# Chest X-Ray Classification

Detect common thoracic diseases from chest X-ray images, using the ChestX-ray8 dataset.

## XAI demos

There are some jupyter notebooks with prediction samples alongside a heatmap over the image, to _explain_ the prediction.

* [CAM examples](Testing.ipynb)

* [LIME examples](Demo_xai.ipynb)

* [IntegratedGradients and DeepLIFT examples](Demo_captum.ipynb), using the [captum](https://captum.ai/) library.


---

## Usage


### Setup

1. Setup [pytorch](http://pytorch.org/). Tested with pytorch v1.4.0 and torchvision v0.5.0.
2. Download the ChestX-ray8 dataset [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).
3. Extract the images into directory `<folder>/dataset/images`.
4. Put `Data_Entry_2017.csv`, `BBox_List_2017.csv`, `test_list.txt`, and `train_val_list.txt` into directory `<folder>/dataset/`.

To generate train, validation, and test data entry.

    python label_gen.py
 
This will separate `train_val_list.txt` into `train_list.txt` and `val_list.txt`.  
3 csv files `train_label.csv`, `val_label.csv`, and `test_label.csv` will be generated as data entry.

### Training

To train models:
```
    python train.py
```

See `python train.py --help` for hyper-parameter options.



---


## References

* ChestX-ray8 dataset
Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: [Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf), IEEE CVPR, pp. 3462-3471,2017


* CAM
TODO