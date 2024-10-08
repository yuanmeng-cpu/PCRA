# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 16:14:43 2018

@author: truthless
"""
# from SDCR import TopicRNN_GCN
# from TopicRNN_GCN import TopicRNN_GCN
from Transformer_Encoder_Sparse_GraphAttention import TopicRNN_GCN
# from newMultihopSparseTopicRNN import TopicRNN_GCN
# from HRED import TopicRNN_GCN
# from Mem2Seq import TopicRNN_GCN
from TopicOptimizer import loss_function
from DataManager import DataManager
from Parser import getParser
from Metrics import bleu, bleu_rectified, bleu_corpus, bleu_str, bleu_str_rectified, bleu_str_corpus, language_model_p, perplexity, cal_acc, cal_acc_new, get_entity, NDCG, getNDCG, distinct_metrics, getMRR
import sys, os
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from tensorboard_logger import configure, log_value
from datetime import datetime
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train():
    writer = SummaryWriter()
    argv = sys.argv[1:]
    parser = getParser()
    args, _ = parser.parse_known_args(argv)
    print(args)
    
    if not os.path.exists(args.save):
        os.mkdir(args.save)
        
    start_timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_output = open("logs/log_%s_%s.txt" %(args.model, start_timestamp), "w")
    log_output.write('{0}\n'.format(str(args)))
    
    torch.manual_seed(args.seed)
    
    manager = DataManager(args.stopword_freq_lb, args.data, args.no_wordvec, args.embed, args.context_len)
    train = manager.create_dataset('train', args.batch)
    valid = manager.create_dataset('valid', args.batch)
    test = manager.create_dataset('test', args.batch)
    
    nodes_rep = manager.nodes_rep
    adj = manager.adj
    
    model = TopicRNN_GCN(args, args.model, args.rnn, manager.vector, manager.word2index, len(manager.word2index), len(manager.index2nonstop), args.embed, args.rnn_dim, args.infer_dim, args.topic, manager.n_entity, nodes_rep, adj, gcn_dropout=0.5, teacher_forcing=args.teacher)
    model.to(device=device)
    if args.load_pretrain:
        pretrain_model = torch.load(args.load) 
        model_dict = model.state_dict() 
        pretrained_dict = pretrain_model.state_dict() 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict) 
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # optimizer = optim.Adam([{'params':model.parameters()},
    #                         {'params':model.gc1.parameters(),'lr':0.1},
    #                         {'params':model.gc1.parameters(),'lr':0.1}],lr = args.lr)

    # initialize and configure tensorboard
    configure(os.path.join(args.save, start_timestamp), flush_secs=5)
    
    best_bleu_score_ori = .0
    best_perplexity_score = .0
    best_entity_accuracy = .0
    
    for epoch in range(args.epoch):
        num = 0
        model.train()
        total_loss, CE_loss, KLD_loss, SCE_loss, ENTCE_loss = 0., 0., 0., 0., 0.
        for batch_cnt, data in enumerate(train):
            model.train()
            optimizer.zero_grad()
            outputs, word_p, indicator_p, mu, logvar, entity_p = model(data[0], data[1], data[2], data[3], data[5])
            entity_grd = data[6]
            entity_mask = data[7]
            CE, KLD, SCE, ENTCE = loss_function(word_p, data[2], indicator_p, data[4], mu, logvar, data[3], entity_p, entity_grd, entity_mask)
        
            # without gcn entity prediction
            if args.model == "TopicRNN":
                loss = CE + KLD + SCE
            # with gcn entity prediction
            else:
                loss = CE + KLD + SCE + 0.6 * ENTCE
    
            loss.backward()

    
            total_loss += loss.item()
            CE_loss += CE.item()
            KLD_loss += KLD.item()
            SCE_loss += SCE.item()
            ENTCE_loss += ENTCE.item()
            optimizer.step()
    
            step = int(batch_cnt + epoch*len(train) + 1)
            log_value("total_loss", loss.item(), step)
            log_value("CE", CE.item(), step)
            log_value("KLD", KLD.item(), step)
            log_value("SCE", SCE.item(), step)
            log_value("ENTSCE", ENTCE.item(), step)
    
            if (batch_cnt+1) % args.print_per_batch == 0:
                total_loss /= args.print_per_batch
                CE_loss /= args.print_per_batch
                KLD_loss /= args.print_per_batch
                SCE_loss /= args.print_per_batch
                ENTCE_loss /= args.print_per_batch
                info = '====> Epoch: {}, Iteration: {}, total_loss:{:.4f}, CE:{:.4f}, KLD:{:.4f}, SCE:{:.4f}, ENTCE:{:.4f}'.format(epoch, batch_cnt, total_loss, CE_loss, KLD_loss, SCE_loss, ENTCE_loss)
                print(info)
                # writer.add_scalar('edge_num/0',num,epoch)
                # print('epoch:{0}/num:{1}'.format(epoch, num))
                log_output.write(info + "\n")

                total_loss, CE_loss, KLD_loss, SCE_loss, ENTCE_loss = 0., 0., 0., 0., 0.



        #validate_model(model, valid, epoch, log_output)
        # print(model.edge_number)
        _, _, _, _, bleu_score, bleu_score_ori, bleu_score_corpus, bleu_score_ori_corpus, perplexity_score, accuracy, dis_1, dis_2, dis_3, dis_4 = test_model(model.to(device=device), valid, epoch, manager, log_output, manager.id_nodename_map, "valid")
        #all_input_sent, all_trg_seqs_ori, all_outputs_ori, all_ent_grd_truth, _, _, _, _, _, _ = test_model(model, test, epoch, manager, log_output, manager.id_nodename_map, "test")
        all_input_sent, all_trg_seqs_ori, all_outputs_ori, all_ent_grd_truth, bleu_score, bleu_score_ori, bleu_score_corpus, bleu_score_ori_corpus, perplexity_score, accuracy, dis_1, dis_2, dis_3, dis_4 = test_model(model, test, epoch, manager, log_output, manager.id_nodename_map, "test")
    
        if bleu_score_ori > best_bleu_score_ori:
            best_bleu_score_ori = bleu_score_ori
            info = "Valid achieve best BLEU score: %f" %(best_bleu_score_ori)
            print(info)
            log_output.write(info + "\n")
    
            model_path = "{0}/model_{1}_best_bleu".format(args.save, args.model)
            print("Save model: ", model_path)
            torch.save(model, model_path)
            json.dump(all_input_sent, open("results/%s/best_bleu_test_input_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_trg_seqs_ori, open("results/%s/best_bleu_test_trg_seqs_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_outputs_ori, open("results/%s/best_bleu_test_output_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_ent_grd_truth, open("results/%s/best_bleu_test_trg_ents_%s.json" %(args.model, start_timestamp), "w"))
            print("Save results for best bleu")
            

        if accuracy > best_entity_accuracy:
            best_entity_accuracy = accuracy 
            info = "Valid achieve best Recall: %f" %(best_entity_accuracy)
            print(info)
            log_output.write(info + "\n")
    
            model_path = "{0}/model_{1}_best_recall".format(args.save, args.model)
            print("Save model: ", model_path)
            torch.save(model, model_path)
            json.dump(all_input_sent, open("results/%s/best_recall_test_input_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_trg_seqs_ori, open("results/%s/best_recall_test_trg_seqs_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_outputs_ori, open("results/%s/best_recall_test_output_%s.json" %(args.model, start_timestamp), "w"))
            json.dump(all_ent_grd_truth, open("results/%s/best_recall_test_trg_ents_%s.json" %(args.model, start_timestamp), "w"))
            print("Save results for best recall")

            

def test_model(model, test, epoch, data_manager, log_output, id_nodename_map, dataset="test"):
    model.eval()
    L, p, N, accuracy = 0, 0, 0, 0
    ttl_ents, correct_ents = 0, 0
    recalls,maps,ndcgs,mrrs = 0,0,0,0
    # dis_1, dis_2, dis_3, dis_4 = 0, 0, 0, 0

    bleu_score, bleu_score_ori = [], []
    bleu_score_len1, bleu_score_ori_len1 = [], []
    all_input_sent = []
    all_ent_grd_truth = []
    all_ent_pred_res = []

    all_trg_seqs = []
    all_outputs = [] 

    all_trg_seqs_ori = []
    all_outputs_ori = [] 
    with torch.no_grad():
        for i, data in enumerate(test):
            # print(i)
          # if i <= 1:
          #   print(i)
            outputs, word_p, indicator_p, mu, logvar, entity_p = model(data[0], data[1], data[2],
                                                                 data[3], data[5],
                                                                 use_teacher_forcing=False)

            trg_seqs = data[2].cpu().numpy().tolist()
            output = outputs.cpu().numpy().tolist()
            tmp_score, tmp_len1_scores = bleu(trg_seqs, output)
            bleu_score += tmp_score

            #dis_1, dis_2, dis_3, dis_4 = distinct_metrics(output)
            
            bleu_score_len1 += tmp_len1_scores
            #bleu_score += bleu_rectified(trg_seqs, output)
            all_trg_seqs += trg_seqs 
            all_outputs += output

            # post process to:
            # 1. replace <$> with its real entity string
            # 2. replace address/phone/postcode with its real postcode
            trg_seqs_ori = data[8]
            outputs_ori = data_manager.post_process(outputs, entity_p)
            tmp_scores, tmp_len1_scores = bleu_str(trg_seqs_ori, outputs_ori)
            bleu_score_ori += tmp_scores
            bleu_score_ori_len1 += tmp_len1_scores
            #bleu_score_ori += bleu_str_rectified(trg_seqs_ori, outputs_ori)
            all_trg_seqs_ori += trg_seqs_ori
            all_outputs_ori += outputs_ori

            entity_grd = data[6]
            #获取推荐列表
            rec_list, rec_list_index = data_manager.get_recommendation_list(outputs, entity_p)

            # print((rec_list)) #64
            entity_list, entities = get_entity(entity_grd, trg_seqs_ori,outputs_ori)
            # print(entities) #64
            '''
            nodename_id_map = {value:key for key, value in id_nodename_map.items()}
            entity_index_list = []
            for entity in entity_list:
                if entity in nodename_id_map.keys():
                    id = nodename_id_map[entity]
                    entity_index_list.append(id)
            '''
            #recall
            recall_sum = 0
            for entity,rec_sublist in zip(entities, rec_list):
                k = 0
                # print("entity:{0},rec_sublist:{1}".format(entity, rec_sublist))
                if len(rec_sublist) != 0:
                    for item in rec_sublist:
                        if item in entity:
                            k += 1
                            # print("k:{0}".format(k))
                            # print("len(rec_sublist):{0}".format(len(rec_sublist)))
                    recall = k / len(rec_sublist)
                    # print("recall:{0}".format(recall))
                    recall_sum += recall
                    # print("recall_sum:{0}".format(recall_sum))
            aver_recall_sum = recall_sum / len(rec_list)
            # print("aver_recall_sum:{0}".format(aver_recall_sum))
            recalls += aver_recall_sum  #recalls代表一个batch_size大小测试样本的平均召回率
            # print("recalls:{0}".format(recalls))
            #MAP
            ap_sum = 0
            for entity, rec_sublist in zip(entities, rec_list):
                # print(entity, rec_sublist)
                hits = 0
                sum_precs = 0
                if len(rec_sublist) != 0:
                    for n in range(len(rec_sublist)):
                        if rec_sublist[n] in entity:
                            hits += 1
                            # print('hits:{0}'.format(hits))
                            sum_precs +=  hits / (n + 1.0)
                            # print('sum_precs:{0}'.format(sum_precs))
                    if hits > 0:
                        ap = sum_precs / len(rec_sublist)
                        # print('ap:{0}'.format(ap))
                        ap_sum += ap
            aver_ap_sum = ap_sum / len(rec_list)
                    # print(aver_ap_sum)
            maps  += aver_ap_sum
            #ndcg
            ndcg_sum = 0
            mrr_sum = 0
            for entity, rec_sublist in zip(entities, rec_list):
                sub_ndcg = 0
                sub_mrr = 0
                if len(rec_sublist) != 0:
                    for item in entity:
                        ndcg = getNDCG(rec_sublist,item)
                        sub_ndcg  += ndcg
                        mrr = getMRR(rec_sublist,item)
                        sub_mrr += mrr
                    ndcg_sum += sub_ndcg
                    mrr_sum += sub_mrr
            aver_ndcg_sum = ndcg_sum / len(rec_list)
            ndcgs += aver_ndcg_sum
            aver_mrr_sum = mrr_sum / len(rec_list)
            mrrs += aver_mrr_sum
            '''   
            print(recalls)
            exit()
            '''

            #ttl_ent, correct_ent, ent_pred_res = cal_acc(entity_p, entity_grd)
            ttl_ent, correct_ent = cal_acc_new(entity_grd, trg_seqs_ori, outputs_ori)
            ttl_ents += ttl_ent
            correct_ents += correct_ent
            
            all_input_sent += data[0].cpu().numpy().tolist()
            all_ent_grd_truth += entity_grd.cpu().numpy().tolist()

            L += len(data[3])
            outputs, word_p, indicator_p, mu, logvar, entity_p = model(data[0], data[1], data[2],
                                                                 data[3], data[5],
                                                                 use_teacher_forcing=True)
            p += language_model_p(data[2], word_p, data[3])
            #data[2]: trg_sequences, word_p, trg_lens
            N += sum(data[3])

    #json.dump(all_input_sent, open("./%s_input_sequence.json" %(dataset), "w"))
    #json.dump(all_ent_grd_truth, open("./%s_entity_ground_truth.json" %(dataset), "w"))



    recall_model = recalls / 115  #115测试总batch(64)数量
    map_model = maps / 115
    ndcg_model = ndcgs / 115
    mrr_model = mrrs / 115
    # print(recall_model,map_model,ndcg_model)
    accuracy = correct_ents / float(ttl_ents)
    info = "total entities: %d, correct entities: %d" %(ttl_ents, correct_ents)
    print(info)
    log_output.write(info + "\n")
    bleu_score = sum(bleu_score) / len(bleu_score)

    len1_number = len(bleu_score_len1)
    if len1_number == 0:
        bleu_score_len1 = 0
    else:
        bleu_score_len1 = sum(bleu_score_len1) / len(bleu_score_len1)
    bleu_score_ori = sum(bleu_score_ori) / len(bleu_score_ori)
    if len1_number == 0:
        bleu_score_ori_len1 = 0
    else:
        bleu_score_ori_len1 = sum(bleu_score_ori_len1) / len(bleu_score_ori_len1)
    perplexity_score = perplexity(p, N)

    dis_1, dis_2, dis_3, dis_4 = distinct_metrics(all_outputs)

    bleu_score_corpus = bleu_corpus(all_trg_seqs, all_outputs)
    bleu_score_ori_corpus = bleu_str_corpus(all_trg_seqs_ori, all_outputs_ori)

    info = '%s set, number of len1 hyp: %d,  Perplexity: %f, Accuracy: %f, Recall:%f, Map:%f, NDCG：%f, MRR: %f, dist-1:%f, dist-2:%f, dist-3:%f, dist-4:%f' %(dataset, len1_number, perplexity_score, accuracy, recall_model,map_model, ndcg_model, mrr_model, dis_1, dis_2, dis_3, dis_4)
    print(info)
    log_output.write(info + "\n")


    return all_input_sent, all_trg_seqs_ori, all_outputs_ori, all_ent_grd_truth, perplexity_score, accuracy, dis_1, dis_2, dis_3, dis_4


def just_test():
    argv = sys.argv[1:]
    parser = getParser()
    args, _ = parser.parse_known_args(argv)
    print(args)

    start_timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_output = open("logs/log_%s_%s.txt" % (args.model, start_timestamp), "w")
    log_output.write('{0}\n'.format(str(args)))
    
    torch.manual_seed(args.seed)
    
    manager = DataManager(args.stopword_freq_lb,args.data, args.no_wordvec, args.embed, args.context_len)
    valid = manager.create_dataset('valid', args.batch)
    test = manager.create_dataset('test', args.batch)
    
    nodes_rep = manager.nodes_rep
    adj = manager.adj

    model = TopicRNN_GCN(args, args.model, args.rnn, manager.vector, manager.word2index, len(manager.word2index), len(manager.index2nonstop), args.embed, args.rnn_dim, args.infer_dim, args.topic, manager.n_entity, nodes_rep, adj, gcn_dropout=0.5, teacher_forcing=args.teacher)
    model.to(device=device)

    pretrain_model = torch.load("./checkpoints/model_gcn_no_activation_best_recall")
    model_dict = model.state_dict() 
    pretrained_dict = pretrain_model.state_dict() 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    # bleu_score, bleu_score_ori, bleu_score_corpus, bleu_score_ori_corpus, perplexity_score, accuracy = test_model(model, test, 0, manager)
    all_input_sent, all_trg_seqs_ori, all_outputs_ori, all_ent_grd_truth,  perplexity_score, accuracy, dis_1, dis_2, dis_3, dis_4 = \
    test_model(model, test, 0, manager, log_output, manager.id_nodename_map, "test")

    return perplexity_score, accuracy, dis_1, dis_2, dis_3, dis_4

if __name__ == "__main__":
    #train()
    just_test()
