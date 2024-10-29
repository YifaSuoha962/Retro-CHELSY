##  Diverse Retrosynthesis via Hierarchical Latent Variables and Chemical Synonyms
## Reproduce the Results
### 1. Environmental setup
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. 
```
conda create -n retrog2s python==3.9.19
conda activate retrog2s
pip install -r requirements.txt
```

### 2. Preprocess Reaction Data
Preprocess the csv file to batch.
```
python preprocess.py --dataset_name $DATASET
```
$DATASET can be selected from[**uspto_50k**, **uspto_diverse**, **uspto_full**].

### 3. Model training and validation
#### 3.1 Taining
Here is the training command and descriptions for each argument:
```
export CUDA_VISIBLE_DEVICES=1 # set id of CUDA device, 0 in default

python train.py 
--dataset_name          # Dataset                            [uspto_50k, uspto_diverse, uspto_full]
--model_type            # model_name                         ['BiG2S', 'BiG2S_HCVAE', 'S2S', 'S2S_HCVAE']
--beam_module           # beam_search_module                 ['OpenNMT', 'huggingface'] 
--train_task            # the prediction task to perform (forward reaction prediction/Retrosynthesis/both)    
                                                             ['prod2subs', 'subs2prod', 'bidirection']
--loss_type             # training loss                      ['CE', 'Mixed-CE', 'focal']   
--lat_disc_size         # size of latent variable that means range of reaction types    
--representation_form   # how to represent product and reactants in translation 
                                                             ['graph2smiles', 'smiles2smiles']                                        
```
Optionally, one can run the demo command:  
```
python train.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE --beam_module OpenNMT --train_task prod2subs --loss_type CE --lat_disc_size 90 --representation_form graph2smiles
```
#### 3.2 Evaluation
Run the `predict.py` with the same arguments in train command, for example:
```
python predict.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE --beam_module OpenNMT --train_task prod2subs --loss_type CE --lat_disc_size 90 --representation_form graph2smiles
```
The results of Top-k Acc, Coverage rate will be saved in the directory ’checkpoints‘, the information of predictions will be saved in directory 'results'.
#### 3.3 Round-Trip Experiment
One who want to perform round-trip experiment, should install [Mol Transformer](https://github.com/pschwllr/MolecularTransformer) previously.
After installation, run the command:
```
python round_trip.py --dataset_name uspto_diverse --result_file <copy the path wanted csv in directory `results`>
```
The results will be saved in directory 'round_trip'.
#### 3.4 Inference for a Chosen Molecule
To perform retrosynthesis or forward reaction prediction for arbitrary molecules, one replace the string in `input_smi` int python file `inference.py` and run the command:  
```
python inference.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE --beam_module OpenNMT --train_task prod2subs --loss_type CE --lat_disc_size 90 --representation_form graph2smiles
```

## Acknowledgement
We refer to the codes of [Graph2SMILES](https://github.com/coleygroup/Graph2SMILES), [RetroDCVAE](https://github.com/MIRALab-USTC/DD-RetroDCVAE), [BiG2S](https://github.com/AILBC/BiG2S) and [RetroBridge](https://github.com/igashov/RetroBridge). Thanks for their contributions.  
