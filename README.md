```
python train_manual_belly_unet.py --images-dir data/source/ribbed_newt_data-fixed --masks-dir masks_manual/ribbed_newt_data-labeled-fixed/ --epochs 40 
--batch-size 6 --output output/ribbed.pt
```

```
python model/train_manual_belly_unet.py --images-dir model/data/source/all_dataset_1_newts_fixed/ --masks-dir model/masks_manual/all_dataset_1_masks_fixed/ --epochs 40 
--batch-size 6 --output output/segmentation_all_kinds.pt
```

```
python train_manual_belly_unet.py --images-dir data/source/karelin_newt_data-fixed/ --masks-dir masks_manual/karelin_newt_labeled-fixed/ --epochs 40 --batch-size 6 --output output/karelin_2.pt
```

```
python3 predict_manual_belly_unet.py --images-dir data/source/karelin_newt_data/ --checkpoint output/belly_unet_manual.pt --output-dir output/test_belly_masks_karelin_run2
````