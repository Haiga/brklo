# activate venv and set Python path
source venv/bin/activate
export PYTHONPATH=PYTHONPATH:$pwd

data=MSMARCO_RETRIEVING
model=RetrieverBERT
max_epochs=1
patience=1
loss=NTXentLoss

for fold_idx in $(seq 4 4);
do
    # fit
    time_start=$(date '+%Y-%m-%d %H:%M:%S')
    python main.py \
      tasks=[fit] \
      trainer.max_epochs=$max_epochs \
      trainer.patience=$patience \
      model=$model \
      model.loss.params.name=$loss \
      data=$data \
      data.folds=[$fold_idx]
    time_end=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$time_start,$time_end" > resource/time/${model}_${data}_fit_${fold_idx}.tmr
#
#    # predict
#    time_start=$(date '+%Y-%m-%d %H:%M:%S')
#    python main.py \
#      tasks=[predict] \
#      model=$model \
#      data=$data \
#      data.folds=[$fold_idx]
#    time_end=$(date '+%Y-%m-%d %H:%M:%S')
#    echo "$time_start,$time_end" > resource/time/${model}_${data}_predict_${fold_idx}.tmr
#
#    # eval
#    time_start=$(date '+%Y-%m-%d %H:%M:%S')
#    python main.py \
#      tasks=[eval] \
#      model=$model \
#      data=$data \
#      data.folds=[$fold_idx]
#    time_end=$(date '+%Y-%m-%d %H:%M:%S')
#    echo "$time_start,$time_end" > resource/time/${model}_${data}_eval_${fold_idx}.tmr
done

