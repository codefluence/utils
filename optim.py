import torch
import numpy as np

from data import JaneStreetData
from model import utility_score
from tabnet import JaneStreetTabNetClassifier, MyBatchSampler

from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.metrics import Metric
from matplotlib import pyplot as plt
import os
import gc
import datetime

import optuna

def optimize(self, data):

    X_train, y_train, c_train = data.get_train_input()
    X_val, y_val, c_val = data.get_val_input()

    def objective(trial):

        class OptunaCallback(Callback):
        
            def on_train_begin(self, logs=None):

                print('trial.params:',trial.params)

            def on_epoch_end(self, epoch, logs=None):

                trial.report(logs['val_u'], epoch)

                # h = clf.history['val_u']
                # if len(h) > 4 and len(h) < 11:

                #     print('h[-1] - h[-5] is ', h[-1] - h[-5])

                #     if (h[-1] - h[-5]) < 3:
                #         print('too slow, giving up..')
                #         raise optuna.exceptions.TrialPruned()

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        #n_d = trial.suggest_int('layer_width', 20, 52)
        #n_a = trial.suggest_int('layer_embedding', 20, 52)
        #n_steps = trial.suggest_int('n_steps', 3, 6)
        #gamma = trial.suggest_uniform('gamma', 1., 2.)
        #momentum = trial.suggest_uniform("momentum", 0.01, 0.4)
        #lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-4, log=True)
        #mask_type = trial.suggest_categorical('mask_type', ['entmax', 'sparsemax'])

        batch_size = trial.suggest_int('batch_size', 5000, 30000)
        n_independent = trial.suggest_int('n_independent', 2, 4)
        n_shared = trial.suggest_int('n_shared', 2, 4)
        lr = trial.suggest_float("lr", 5e-3, 5e-1, log=True)

        clf = JaneStreetTabNetClassifier(n_d=100,
                            n_a=100,
                            n_steps=5,
                            gamma=1.5,
                            n_independent=n_independent,
                            n_shared=n_shared,
                            momentum=0.3,
                            clip_value=2.,
                            lambda_sparse=1e-5,
                            cat_idxs=[0],
                            cat_dims=[2],
                            cat_emb_dim=1,
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=lr),
                            scheduler_params={"step_size":5, "gamma":0.9},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                            mask_type='entmax'
                            )
                    
        torch.manual_seed(0)
        np.random.seed(0)

        #ride = np.random.choice(np.arange(X_train.shape[0]), size=X_val.shape[0], replace=False)

        clf.fit(
            X_train=X_train,
            y_train=y_train,
            #eval_set=[(X_train[ride], y_train[ride]), (X_val, y_val)],
            eval_set=[(X_val, y_val)],
            #eval_name=['train', 'val'],
            eval_name=['val'],
            loss_fn = loss_fn,
            eval_metric=[t_metric, p_metric, u_metric],
            batch_size=batch_size,
            virtual_batch_size=batch_size//8,
            max_epochs=30,
            patience=5,
            #num_workers=4,
            weights=1,
            drop_last=False,
            context_fit=c_train,
            #context_eval_set=[c_train[ride], c_val],
            context_eval_set=[c_val],
            callbacks=[OptunaCallback()]
        )

        best_u = max(clf.history['val_u'])

        saved_filepath = clf.save_model(str(str(round(best_u,2)).zfill(8)) + '-' + \
                        str(trial._study_id) + str(trial._trial_id) + str(trial.params).replace(':', '') + \
                        datetime.datetime.now().strftime("-%d-%m-%Y-%H%M%S"))
        print('Parameters saved in file', saved_filepath)

        return best_u

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=18, callbacks=[lambda study, trial: gc.collect()])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    gc.collect()
    torch.cuda.empty_cache()


    