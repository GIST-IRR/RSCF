#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import operator

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False, args=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.r_weight = args.r_weight
        self.hp = args.hp
        self.rp = args.rp
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        
        self.relation_embedding = nn.Parameter(torch.ones(nrelation, self.relation_dim))
        
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
            )
        
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
            )
        
        self.trans_dim = self.hidden_dim*2 if args.Linear_2 else self.hidden_dim
        
        self.trans_dim = self.trans_dim*2 if double_relation_embedding else self.trans_dim
        
        self.trans_head = nn.Parameter(torch.zeros(self.relation_dim, self.trans_dim))
        self.trans_tail = nn.Parameter(torch.zeros(self.relation_dim, self.trans_dim))
        
        nn.init.xavier_uniform_(self.trans_head)
        nn.init.xavier_uniform_(self.trans_tail)
        
        self.trans_rel_head = nn.Parameter(torch.zeros(self.entity_dim, self.relation_dim))
        self.trans_rel_tail = nn.Parameter(torch.zeros(self.entity_dim, self.relation_dim))
        
        nn.init.xavier_uniform_(self.trans_rel_head)
        nn.init.xavier_uniform_(self.trans_rel_tail)
        
        self.loss_rel = nn.CrossEntropyLoss(reduction='mean')

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, args, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''
        head, relation, tail = self.embedding(sample, mode)
        
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        model_func_rel = {
            'TransE': self.TransE_rel,
            'RotatE': self.RotatE_rel
        }
        
        if mode == 'single':
            score_rel = model_func_rel[self.model_name](head, relation, tail, mode)
            
        trans_head = torch.matmul(relation, self.trans_head)
        trans_tail = torch.matmul(relation, self.trans_tail)
        
        trans_head = F.normalize(trans_head, dim=2, p=args.hp) + 1
        trans_tail = F.normalize(trans_tail, dim=2, p=args.hp) + 1
    
        trans_rel_head = torch.matmul(head, self.trans_rel_head)
        trans_rel_tail = torch.matmul(tail, self.trans_rel_tail)
        
        trans_rel_head = F.normalize(trans_rel_head, dim=2, p=args.rp) + 1
        trans_rel_tail = F.normalize(trans_rel_tail, dim=2, p=args.rp) + 1
        
        relation = relation * trans_rel_head * trans_rel_tail
        
        if self.model_name == 'RotatE':
            if args.Linear_2:
                head, tail = self.rotate_linear_2(head, tail, trans_head, trans_tail)
            else:
                head, tail = self.rotate_diag(head, tail, trans_head, trans_tail)

        elif self.model_name == 'TransE':
            if args.Linear_2:
                head, tail = self.transe_linear_2(head, tail, trans_head, trans_tail)
            else:
                head, tail = self.transe_diag(head, tail, trans_head, trans_tail)
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
                
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        if mode == 'single':
            return score, score_rel
        return score

    def rotate_linear_2(self, head, tail, trans_head, trans_tail):
        r_h = torch.chunk(trans_head, 4, dim=2)
        r_t = torch.chunk(trans_tail, 4, dim=2)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_head_1, re_head_2 = torch.chunk(re_head, 2, dim=2)
        im_head_1, im_head_2 = torch.chunk(im_head, 2, dim=2)

        re_tail_1, re_tail_2 = torch.chunk(re_tail, 2, dim=2)
        im_tail_1, im_tail_2 = torch.chunk(im_tail, 2, dim=2)

        re_head_new_1 = re_head_1 * r_h[0] + re_head_2 * r_h[1]
        re_head_new_2 = re_head_1 * r_h[2] + re_head_2 * r_h[3]

        im_head_new_1 = im_head_1 * r_h[0] + im_head_2 * r_h[1]
        im_head_new_2 = im_head_1 * r_h[2] + im_head_2 * r_h[3]

        re_tail_new_1 = re_tail_1 * r_t[0] + re_tail_2 * r_t[1]
        re_tail_new_2 = re_tail_1 * r_t[2] + re_tail_2 * r_t[3]

        im_tail_new_1 = im_tail_1 * r_t[0] + im_tail_2 * r_t[1]
        im_tail_new_2 = im_tail_1 * r_t[2] + im_tail_2 * r_t[3]

        re_head = torch.cat([re_head_new_1, re_head_new_2], dim=2)
        im_head = torch.cat([im_head_new_1, im_head_new_2], dim=2)

        re_tail = torch.cat([re_tail_new_1, re_tail_new_2], dim=2)
        im_tail = torch.cat([im_tail_new_1, im_tail_new_2], dim=2)

        head = torch.cat([re_head, im_head], dim=2)
        tail = torch.cat([re_tail, im_tail], dim=2)

        return head, tail
    
    def rotate_diag(self, head, tail, trans_head, trans_tail):
        head = head * torch.cat([trans_head, trans_head], dim=2)
        tail = tail * torch.cat([trans_tail, trans_tail], dim=2)
        
        return head, tail

    def transe_linear_2(self, head, tail, trans_head, trans_tail):
        r_h = torch.chunk(trans_head, 4, dim=2)
        r_t = torch.chunk(trans_tail, 4, dim=2)
        head_1, head_2 = torch.chunk(head, 2, dim=2)
        tail_1, tail_2 = torch.chunk(tail, 2, dim=2)

        head_new_1 = head_1 * r_h[0] + head_2 * r_h[1]
        head_new_2 = head_1 * r_h[2] + head_2 * r_h[3]

        tail_new_1 = tail_1 * r_t[0] + tail_2 * r_t[1]
        tail_new_2 = tail_1 * r_t[2] + tail_2 * r_t[3]

        head = torch.cat([head_new_1, head_new_2], dim=2)
        tail = torch.cat([tail_new_1, tail_new_2], dim=2)
            
        return head, tail

    def transe_diag(self, head, tail, trans_head, trans_tail):
        head, tail = head * trans_head, tail * trans_tail
        return head, tail
    
    def embedding(self, sample, mode):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        return head, relation, tail

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score

    def TransE_rel(self, head, relation, tail, mode):
        if mode == 'single':
            score_rel = (head + self.relation_embedding) - tail
            score_rel = self.gamma.item() - torch.norm(score_rel, p=1, dim=2)
            return score_rel
        
        
    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)
        score = self.gamma.item() - score.sum(dim = 2)
    
        return score
    
    def RotatE_rel(self, head, relation, tail, mode):
        
        if mode == 'single':
            pi = 3.14159265358979323846
            
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)
            
            phase_relation_all = self.relation_embedding/(self.embedding_range.item()/pi)
            re_relation_all = torch.cos(phase_relation_all)
            im_relation_all = torch.sin(phase_relation_all)
            
            re_score_rel = re_head * re_relation_all - im_head * im_relation_all
            im_score_rel = re_head * im_relation_all + im_head * re_relation_all
            re_score_rel = re_score_rel - re_tail
            im_score_rel = im_score_rel - im_tail
            
            score_rel = torch.stack([re_score_rel, im_score_rel], dim = 0)
            score_rel = score_rel.norm(dim = 0)
            score_rel = self.gamma.item() - score_rel.sum(dim = 2)
            
            return score_rel
        
        
    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), args, mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score, rel_score = model(positive_sample, args)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        rel_loss = model.loss_rel(rel_score, positive_sample[:, 1])
        rel_loss *= model.r_weight
        loss += rel_loss
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'rel_loss': rel_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample, args).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), args, mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics