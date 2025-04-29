# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook,DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .utils import udwMcDropoutHook,GCELoss


@ALGORITHMS.register('udw')

class UDW(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff,kappa_p=args.kappa_p, loss_cosine_decay_epoch=args.loss_cosine_decay_epoch,hard_label=args.hard_label)
    
    def init(self, T, p_cutoff,kappa_p,loss_cosine_decay_epoch, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.kappa_p=kappa_p
        self.gceloss = GCELoss(q=0.8)
        self.loss_cosine_decay_epoch = loss_cosine_decay_epoch
        self.flag=False
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(udwMcDropoutHook(num_classes=self.num_classes,n_sigma=self.args.n_sigma, momentum=self.args.ema_p, per_class=self.args.per_class), "udwMcDropoutHook")
        # self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='uniform' if self.args.dist_uniform else 'model'), 
            "DistAlignHook")

        super().set_hooks()

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        from semilearn.datasets.cv_datasets.starfield import get_custom_dataset
        lb_dset, ulb_dset, eval_dset = get_custom_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels, self.args.num_classes, data_dir=self.args.data_dir, include_lb_to_ulb=self.args.include_lb_to_ulb)
        test_dset = None
        dataset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset, 'test': test_dset}
        
        if dataset_dict is None:
            return dataset_dict
        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        # inference and calculate sup/unsup losses
        # print(self.use_cat)
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}



            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # print(self.epoch)


            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())


            # compute mask
            mask1,mask2 = self.call_hook("masking", "udwMcDropoutHook",probs_x_ulb_w=probs_x_ulb_w,inputs=x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 

                                        logits=logits_x_ulb_w,
                                        use_hard_label=self.use_hard_label,
                                        T=self.T)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                            pseudo_label,
                                            'ce',
                                            mask=mask1)+0.5*self.gceloss(logits_x_ulb_s,pseudo_label,mask2)
        

            total_loss = sup_loss + self.lambda_u * unsup_loss
            assert not torch.isnan(x_lb).any(), "x_lb Input data contains NaN!"
            assert not torch.isinf(x_lb).any(), "x_lb Input data contains infinity!"
            assert not torch.isnan(y_lb).any(), "x_lb Labels contain NaN!"
            assert not torch.isinf(y_lb).any(), "x_lb Labels contain infinity!"

            assert not torch.isnan(x_ulb_w).any(), "x_lb Input data contains NaN!"
            assert not torch.isinf(x_ulb_w).any(), "x_lb Input data contains infinity!"
            assert not torch.isnan(pseudo_label).any(), "pseudo_label Labels contain NaN!"
            assert not torch.isinf(pseudo_label).any(), "pseudo_label Labels contain infinity!"

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                            unsup_loss=unsup_loss.item(), 
                                            total_loss=total_loss.item(), 
                                            ez1util_ratio=mask1.float().mean().item(),ez2util_ratio=mask2.float().mean().item())
        return out_dict, log_dict
    def warm_train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        # inference and calculate sup/unsup losses
        # print(self.use_cat)
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}



            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            # print(self.epoch)


            total_loss = sup_loss

            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item())

        return out_dict, log_dict
    
    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            self.call_hook("before_train_epoch")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                if epoch<15:
                    self.out_dict, self.log_dict = self.warm_train_step(**self.process_batch(**data_lb, **data_ulb))
                else:
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))

                self.call_hook("after_train_step")
                self.it += 1
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
    
    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--kappa_p', float, 0.35),
            SSL_Argument('--p_cutoff', float, 0.9),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--n_sigma', int, 2),
            SSL_Argument('--per_class', str2bool, False),
            SSL_Argument('--loss_cosine_decay_epoch', int, 100),

        ]