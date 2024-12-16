"""#############################################################################
#################### SPE IT MODEL TRAINING #####################################
author : Chabrun Floris and Dieu Xavier
date : 22/11/2024
Training segmentation models for immunosubtraction data
#############################################################################"""
# general modules
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json

# PYTORCH
import torch
import torch.utils.data as data
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# %%

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # default, need to be here
    parser.add_argument("--host")  # dummy arg => for PyCharm only
    parser.add_argument("--port")  # dummy arg => for PyCharm only
    parser.add_argument("--mode")  # dummy arg => for PyCharm only
    parser.add_argument("--run_mode", type=str, default="auto")  # run_mode=auto will set to training+validation if no ckpt is found, else validation
    parser.add_argument("--model_name", type=str,
                        default="default_segformer")  # name of the model, will be used to 1) load the right config file and 2) export to a custom new folder
    parser.add_argument("--data_root_path", type=str,
                        default=None)  # path in which look for the config file // if None, will try to see if local (Floris' or Xavier's PC)
    parser.add_argument("--config_root_path", type=str,
                        default=None)  # path in which look for the config file // if None, will try to see if local (Floris' or Xavier's PC)
    parser.add_argument("--output_root_path", type=str,
                        default=None)  # path in which look for the config file // if None, will try to see if local (Floris' or Xavier's PC)
    parser.add_argument("--dependencies_path", type=str, default=None)  # if a path has to be added to sys path before loading custom dependencies
    parser.add_argument("--debug", type=str,
                        default="file")  # set to None for no plots; set to "inline" to plt.show() debug plots; set to anything else to plt.save() debug plots to this path
    parser.add_argument("--num_workers", type=int, default=0)  # how many workers // 0 for local PC (1, no parallelization)

    args = parser.parse_args()

    """=============================================================================
    Paths and data loading
    ============================================================================="""

    # default paths, if unspecified
    console_verbose = False
    if (args.config_root_path is None) and (args.data_root_path is None) and (args.output_root_path is None):
        if os.path.exists(r"C:\Users"):
            if "flori" in os.listdir(r"C:\Users"):  # floris
                console_verbose = True
                args.config_root_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\ispectr\configs"
                args.data_root_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\data\proc\lemans_2018"
                args.output_root_path = r"C:\Users\flori\Documents\Home\Research\SPECTR\ISPECTR\output"
            elif "afors" in os.listdir(r"C:\Users"):  # Xavier
                console_verbose = True
                args.config_root_path = None  # TODO put the directory in which you'll put the config files // it should be in the Github!!! (see current Github)
                args.data_root_path = r"C:\Users\afors\Documents\Projects\SPE_IT\lemans_2018"
                args.output_root_path = r"C:\Users\afors\Documents\Projects\SPE_IT\output"
        elif os.path.exists("/lustre/fswork/projects/rech/ild/uqk67mt/ispectr"):  # jean zay
            args.config_root_path = "/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/scripts/ispectr/configs"  # directly from git
            args.data_root_path = "/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/data"
            args.output_root_path = "/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/output"
            # add some dependency paths
            args.dependencies_path = ["/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/scripts",
                                      "/lustre/fswork/projects/rech/ild/uqk67mt/ispectr/scripts/ispectr/python",
                                      ]
    # check we have a path for everything
    assert (args.config_root_path is not None), f"{args.config_root_path=}"
    assert (args.data_root_path is not None), f"{args.data_root_path=}"
    assert (args.output_root_path is not None), f"{args.output_root_path=}"

    # dependencies
    if args.dependencies_path is not None:
        import sys
        for dep in args.dependencies_path:
            sys.path.insert(0, dep)

    # custom modules
    from spep_assets.spep_data import ISDataset
    from coding_assets.python import config_manager
    from spep_assets.spep_models import SegformerConfig, IsSegformer, pl_IS_model

    # update output_path to include the model name, and create a new folder // same for debug
    args.output_root_path = os.path.join(args.output_root_path, args.model_name)
    os.makedirs(args.output_root_path, exist_ok=True)
    if (args.debug is not None) and (args.debug != "inline"):  # we're supposed to save figures to files => we'll export them at the same location as the checkpoints
        args.debug = args.output_root_path

    # load config
    config = config_manager.load_config_from_json_at_realpath(os.path.join(args.config_root_path, args.model_name + ".json"))
    # config_manager.impute_config_if_missing(config, "optimizer", "Adam")
    # save config in output directory
    config_manager.save_config_to_json(wdir=args.output_root_path, cf=config)

    # determine if loading in training+val or val only mode:
    if args.run_mode == "auto":
        if os.path.exists(os.path.join(args.output_root_path, "last.ckpt")):
            args.run_mode = "valid"  # do not train because auto & model already exists: only validate model
        else:
            args.run_mode = "full"  # train, then validate (no model exists)

    # load Le Mans data

    # load x array
    # data is already normalized between 0-1 and zero-padded to a 304 width
    if_x = np.load(os.path.join(args.data_root_path, "if_v1_x.npy"))

    # load y array
    if_y = np.load(os.path.join(args.data_root_path, "if_v1_y.npy"))

    # note: we should create .h5 files to easily load the data using a data manager if we have a lot of samples!

    if args.debug is not None:
        # show the firts sample of the dataset
        is_tracks = ["ELP", "IgG", "IgA", "IgM", "K", "L"]

        i = 0
        plt.figure(figsize=(12, 12))
        for j in range(6):
            plt.subplot(6, 2, j * 2 + 1)
            sns.lineplot(x=np.arange(304), y=if_x[i, :, j])  # plot data
            if j > 0:  # plot annotation map
                plt.subplot(6, 2, j * 2 + 2)
                sns.lineplot(x=np.arange(304), y=if_y[i, :, j - 1])
        plt.tight_layout()
        if args.debug != "inline":
            plt.savefig(os.path.join(args.debug, "dummy_xy.png"))
            plt.close()
        else:
            plt.show()

    """=============================================================================
    Data splitting and dataloader setup
    ============================================================================="""

    # we'll output the proportion of each class in the dataset (so we'll check if random partitioning works fine)
    print('IgG% : ', round(if_y[..., 0].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2), '\n',
          'IgA% : ', round(if_y[..., 1].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2), '\n',
          'IgM% : ', round(if_y[..., 2].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2), '\n',
          'Kappa% : ', round(if_y[..., 3].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2), '\n',
          'Lambda% : ', round(if_y[..., 4].max(axis=1).sum() / len(if_y[..., 0].max(axis=1)), 2))

    # partition
    if_x_train, if_x_test, if_y_train, if_y_test = train_test_split(if_x, if_y, test_size=.2, random_state=1, shuffle=True,
                                                                    )
    print('IgG% train : ', round(if_y_train[..., 0].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2), '\n',
          'IgA% train : ', round(if_y_train[..., 1].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2), '\n',
          'IgM% train : ', round(if_y_train[..., 2].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2), '\n',
          'Kappa% train : ', round(if_y_train[..., 3].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2), '\n',
          'Lambda% train : ', round(if_y_train[..., 4].max(axis=1).sum() / len(if_y_train[..., 0].max(axis=1)), 2))

    print('IgG% test : ', round(if_y_test[..., 0].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2), '\n',
          'IgA% test : ', round(if_y_test[..., 1].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2), '\n',
          'IgM% test : ', round(if_y_test[..., 2].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2), '\n',
          'Kappa% test : ', round(if_y_test[..., 3].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2), '\n',
          'Lambda% test : ', round(if_y_test[..., 4].max(axis=1).sum() / len(if_y_test[..., 0].max(axis=1)), 2))

    # seems well stratified

    train_dataset = ISDataset(if_x=if_x_train, if_y=if_y_train,
                              smoothing=config.smoothing, normalize=config.online_normalize,
                              coarse_dropout=config.coarse_dropout, permute=config.permutation)

    # create our dataset loader for val data
    test_dataset = ISDataset(if_x=if_x_test, if_y=if_y_test,
                             smoothing=config.smoothing, normalize=config.online_normalize,
                             coarse_dropout=False, permute=None)  # always false during validation

    # create our dataset loader for train data
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # shuffle during training
        drop_last=True,  # drop last batch if incomplete
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
        # if we set >1 loader, we want them to be persistent, i.e. not being instantiated again between each epoch
    )

    validation_loader = data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False
    )

    if args.debug is not None:
        # test permutations
        for permut_i in [0, 1, 2, 4, 6, 8]:
            test_item = train_dataset.__getitem__(permut_i)
            test_x, test_y = test_item
            test_x = test_x.reshape((test_x.shape[-1], 1, test_x.shape[0],))
            n_permuts = len(train_dataset.permutations) if train_dataset.permutations is not None else "none"
            idx_sample = permut_i // n_permuts if train_dataset.permutations is not None else permut_i
            permut = train_dataset.permutations[permut_i % n_permuts] if train_dataset.permutations is not None else "none"

            is_tracks = ["ELP", "IgG", "IgA", "IgM", "K", "L"]

            plt.figure(figsize=(12, 12))
            for j in range(6):
                plt.subplot(6, 2, j * 2 + 1)
                sns.lineplot(x=np.arange(304), y=test_x[:, 0, j])  # plot data
                if j > 0:  # plot annotation map
                    plt.subplot(6, 2, j * 2 + 2)
                    sns.lineplot(x=np.arange(304), y=test_y[:, j - 1])
            plt.suptitle(f"{idx_sample=} {permut=}")
            plt.tight_layout()
            if args.debug != "inline":
                plt.savefig(os.path.join(args.debug, f"batch item {permut_i=}.png"))
                plt.close()
            else:
                plt.show()

        # just to check if the loader works OK
        first_train_batch = next(iter(train_loader))
        x, y = first_train_batch
        print(f"First training batch: {x.shape=} / {y.shape=}")

    """=============================================================================
    Model instantiation
    ============================================================================="""

    # TODO pre-training: predict peak location on SPEP (if parallel model for 6 channels on IT)

    # TODO pré-entraîner modèle sur une autre tâche
    # TODO masquer les pics pour forcer le modèle à regarder autour de l'albumine

    """=============================================================================
    Model training and validation
    ============================================================================="""

    trainer_args = {'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
                    'devices': 'auto',
                    'num_nodes': 1,
                    'strategy': 'auto'}

    # sending our model into pytorch lightning for training and evaluation
    if args.run_mode == "full":
        if config.architecture == "segformer":
            segformer_config = SegformerConfig(**config.architecture_config)
            model_config = config.model_config
            model = pl_IS_model(IsSegformer, segformer_config, **model_config)
        else:
            assert False, f"Unhandled {config.architecture=}"

        # create our trainer that will handle training
        logger = CSVLogger(save_dir=args.output_root_path, name="logs")
        tb_logger = pl.pytorch.loggers.TensorBoardLogger(save_dir=args.output_root_path, name="tb_logs")
        callbacks = [ModelCheckpoint(dirpath=args.output_root_path, save_weights_only=True,
                                     mode=config.callbacks_config["mode"], monitor=config.callbacks_config["monitor"],
                                     save_last=True),
                     EarlyStopping(verbose=console_verbose,
                                   mode=config.callbacks_config["mode"], monitor=config.callbacks_config["monitor"],
                                   min_delta=config.callbacks_config["early_stopping_min_delta"], patience=config.callbacks_config["early_stopping_patience"]),
                     ]

        trainer = pl.Trainer(
            default_root_dir=args.output_root_path,
            **trainer_args,
            max_epochs=config.max_epochs,
            log_every_n_steps=1,
            callbacks=callbacks,
            enable_progress_bar=console_verbose,
            logger=[logger, tb_logger]
        )

        # fit model
        trainer.fit(model, train_loader, validation_loader)

    """=============================================================================
    Validation metrics
    ============================================================================="""

    # reload last = best (hopefully?) checkpoint
    if config.architecture == "segformer":
        model = pl_IS_model.load_from_checkpoint(os.path.join(args.output_root_path, "last.ckpt"))
    else:
        assert False, f"Unhandled {config.architecture=}"

    trainer = pl.Trainer(
        default_root_dir=args.output_root_path,
        **trainer_args,
        enable_progress_bar=console_verbose,
    )

    # predict on validation data
    validation_outputs = trainer.predict(model, dataloaders=validation_loader)
    # note: in pytorch, the output is a list of N elements, N being the number of batches => so we have to convert that to a np array
    validation_preds = torch.cat(validation_outputs).detach().cpu().numpy()

    export_metrics = dict()

    # some general metrics : point precision and IOU
    threshold = .5
    points = np.arange(1, 304 + 1, 1)
    pr = np.zeros((validation_preds.shape[0], 5))
    iou = np.zeros((validation_preds.shape[0], 5))
    for ix in range(validation_preds.shape[0]):
        for dim in range(5):
            gt = if_y_test[ix, :, dim]
            pd_ = (validation_preds[ix, dim, :] > threshold) * 1
            u = np.sum(gt + pd_ > 0)
            i = np.sum(gt + pd_ == 2)
            if np.isfinite(u):
                iou[ix, dim] = i / u
            else:
                iou[ix, dim] = np.nan
            pr[ix, dim] = np.sum(gt == pd_) / 304

    for k in range(iou.shape[1]):
        print("Mean IoU for fraction '{}': {:.2f} +- {:.2f}".format(['G', 'A', 'M', 'k', 'l'][k], np.nanmean(iou[:, k]),
                                                                    np.nanstd(iou[:, k])))
        export_metrics['IoU-{}'.format(['G', 'A', 'M', 'k', 'l'][k])] = np.nanmean(iou[:, k])
    export_metrics['IoU-global'] = np.nanmean(iou)

    for k in range(pr.shape[1]):
        print("Mean accuracy for fraction '{}': {:.2f} +- {:.2f}".format(['G', 'A', 'M', 'k', 'l'][k], np.nanmean(pr[:, k]),
                                                                         np.nanstd(pr[:, k])))


    # a function for plotting
    def plotITPredictions(ix):
        plt.figure(figsize=(14, 10))
        plt.subplot(3, 1, 1)
        # on récupère la class map (binarisée)
        # class_map = y[ix].max(axis=1)
        curve_values = if_x_test[ix, :, :]
        for num, col, lab in zip(range(6), ['black', 'purple', 'pink', 'green', 'red', 'blue'],
                                 ['Ref', 'G', 'A', 'M', 'k', 'l']):
            plt.plot(np.arange(0, 304), curve_values[:, num], '-', color=col, label=lab)
        plt.title('Valid set curve index {}'.format(ix))

        plt.legend()
        # for peak_start, peak_end in zip(np.where(np.diff(class_map)==1)[0]+1, np.where(np.diff(class_map)==-1)[0]+1):
        #     plt.plot(np.arange(peak_start,peak_end), curve_values[peak_start:peak_end], '-', color = 'red')

        # on plot aussi les autres courbes
        plt.subplot(3, 1, 2)
        for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
            plt.plot(np.arange(0, 304) + 1, if_y_test[ix, :, num] / 5 + (4 - num) / 5, '-', color=col, label=lab)
        plt.ylim(-.05, 1.05)
        plt.legend()
        plt.title('Ground truth maps')

        plt.subplot(3, 1, 3)
        for num, col, lab in zip(range(5), ['purple', 'pink', 'green', 'red', 'blue'], ['G', 'A', 'M', 'k', 'l']):
            plt.plot(np.arange(0, 304) + 1, validation_preds[ix, num, :] / 5 + (4 - num) / 5, '-', color=col, label=lab)
        plt.ylim(-.05, 1.05)
        plt.legend()
        plt.title('Predicted maps')
        if args.debug != "inline":
            plt.savefig(os.path.join(args.debug, f"pred_plot_{ix=}.png"))
            plt.close()
        else:
            plt.show()


    if args.debug is not None:
        for idx in [0, 100, 200]:
            plotITPredictions(idx)

    # Calculons pour chaque pic réel/prédit la concordance
    threshold = 0.5  # ou 0.5
    curve_ids = []
    groundtruth_spikes = []
    predicted_spikes = []
    for ix in range(if_x_test.shape[0]):
        flat_gt = np.zeros_like(if_y_test[ix, :, 0])
        for i in range(if_y_test.shape[-1]):
            flat_gt += if_y_test[ix, :, i] * (1 + np.power(2, i))
        gt_starts = []
        gt_ends = []
        prev_v = 0
        for i in range(304):
            if flat_gt[i] != prev_v:  # changed
                # multiple cases:
                # 0 -> non-zero = enter peak
                if prev_v == 0:
                    gt_starts.append(i)
                # non-zero -> 0 = out of peak
                elif flat_gt[i] == 0:
                    gt_ends.append(i)
                # non-zero -> different non-zero = enter other peak
                else:
                    gt_ends.append(i)
                    gt_starts.append(i)
                prev_v = flat_gt[i]

        if len(gt_starts) != len(gt_ends):
            raise Exception('Inconsistent start/end points')

        if len(gt_starts) > 0:
            # TODO => ce type de validation ne prend pas en compte les faux positifs!
            # pour chaque pic, on détecte ce que le modèle a rendu a cet endroit comme type d'Ig
            for pstart, pend in zip(gt_starts, gt_ends):
                gt_ig_denom = ''
                if np.sum(if_y_test[ix, pstart:pend, :3]) > 0:
                    HC_gt = int(np.median(np.argmax(if_y_test[ix, pstart:pend, :3], axis=1)))
                    gt_ig_denom = ['G', 'A', 'M'][HC_gt]
                lC_gt = int(np.median(np.argmax(if_y_test[ix, pstart:pend, 3:], axis=1)))
                gt_ig_denom += ['k', 'l'][lC_gt]

                pred_ig_denom = ''
                if np.sum(validation_preds[ix, :, pstart:pend] > threshold) > 0:  # un pic a été détecté
                    if np.sum(validation_preds[ix, :3, pstart:pend] > threshold) > 0:
                        HC_pred = int(np.median(np.argmax(validation_preds[ix, :3, pstart:pend], axis=0)))
                        pred_ig_denom = ['G', 'A', 'M'][HC_pred]
                    lC_pred = int(np.median(np.argmax(validation_preds[ix, 3:, pstart:pend], axis=0)))
                    pred_ig_denom += ['k', 'l'][lC_pred]
                else:
                    pred_ig_denom = 'none'

                groundtruth_spikes.append(gt_ig_denom)
                predicted_spikes.append(pred_ig_denom)
                curve_ids.append(ix)
        else:
            gt_ig_denom = 'none'
            pred_ig_denom = ''
            if np.sum(validation_preds[ix, :3, :] > threshold) > 0:
                HC_pred = int(np.median(np.argmax(validation_preds[ix, :3, :], axis=0)))
                pred_ig_denom = ['G', 'A', 'M'][HC_pred]
            lC_pred = int(np.median(np.argmax(validation_preds[ix, 3:, :], axis=0)))
            pred_ig_denom += ['k', 'l'][lC_pred]

            groundtruth_spikes.append(gt_ig_denom)
            predicted_spikes.append(pred_ig_denom)
            curve_ids.append(ix)

    conc_df = pd.DataFrame(dict(ix=curve_ids,
                                true=groundtruth_spikes,
                                pred=predicted_spikes))

    print(pd.crosstab(conc_df.true, conc_df.pred))

    print('Global precision: ' + str(round(100 * np.sum(conc_df.true == conc_df.pred) / conc_df.shape[0], 1)))
    for typ in np.unique(conc_df.true):
        subset = conc_df.true == typ
        print('  Precision for type ' + typ + ': ' + str(
            round(100 * np.sum(conc_df.true.loc[subset] == conc_df.pred.loc[subset]) / np.sum(subset), 1)))
        export_metrics['Acc-{}'.format(typ)] = 100 * np.sum(conc_df.true.loc[subset] == conc_df.pred.loc[subset]) / np.sum(
            subset)
    export_metrics['Acc-global'] = 100 * np.sum(conc_df.true == conc_df.pred) / conc_df.shape[0]

    export_metrics['Mistakes-total'] = conc_df.loc[conc_df.true != conc_df.pred, :].shape[0]
    mistakes = conc_df.loc[conc_df.true != conc_df.pred, 'ix'].unique().tolist()
    export_metrics['Mistakes-curves'] = len(mistakes)

    # export predictions
    np.save(os.path.join(args.output_root_path, "validation_preds.npy"), validation_preds)
    # export metrics
    with open(os.path.join(args.output_root_path, "export_metrics.json"), "w") as jsf:
        json.dump(export_metrics, jsf, indent=4)
