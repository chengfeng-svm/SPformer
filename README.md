# **SPformer**
Download the model weights of SPformer-384-GOT and the test results from [Google Drive](https://drive.google.com/drive/folders/13ytqwWLqK79bkGV4S0wuAe0oITyENwp-?usp=drive_link) 

![backbone](pic/backbone.pdf)

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
# **Tracking Result**
Download the model weights of SPformer-384-GOT and the test results from [Google Drive](https://drive.google.com/drive/folders/13ytqwWLqK79bkGV4S0wuAe0oITyENwp-?usp=drive_link) 

![video1](video/output3.gif)
![video2](video/output4.gif)

## Acknowledgments

* Thanks for [PyTracking](https://github.com/visionml/pytracking) Library, [MixFormer](https://github.com/MCG-NJU/MixFormer), which helps us to quickly implement our ideas.

## Citation