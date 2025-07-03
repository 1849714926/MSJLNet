# MSJLNet
## Datasets
* RegDB [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html).
* SYSU-MM01 [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

  * You need to run `python process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

## Training
**Train IFALNet by**

```
python train.py --dataset sysu --gpu 0
```
* `--dataset`: which dataset "sysu", "regdb" or "llcm".
* `--gpu`: which gpu to run.

## Testing
**Test a model by**
```
python test.py --dataset 'sysu' --mode 'all' --resume 'model_path' --tvsearch True --gpu 0 
```
* `--dataset`: which dataset "sysu", "regdb" or "llcm".
* `--mode`: "all" or "indoor" (only for sysu dataset).
* `--resume`: the saved model path.
* `--tvsearch`: whether thermal to visible search True or False (only for regdb dataset).
* `--gpu`: which gpu to use.
