# -*- coding: utf-8 -*-
import os
cudaid = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)

import sys
import time
import numpy as np
import pickle
import copy
import random
from random import shuffle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data as datar
from model import *
from utils_pg import *
from configs import *
from optim import Optim

cfg = DeepmindConfigs()
TRAINING_DATASET_CLS = DeepmindTraining
TESTING_DATASET_CLS = DeepmindTesting

def print_basic_info(modules, consts, options):
    if options["is_debugging"]:
        print("\nWARNING: IN DEBUGGING MODE\n")
    if options["copy"]:
        print("USE COPY MECHANISM")
    if options["coverage"]:
        print("USE COVERAGE MECHANISM")
    if  options["avg_nll"]:
        print("USE AVG NLL as LOSS")
    else:
        print("USE NLL as LOSS")
    if options["has_learnable_w2v"]:
        print("USE LEARNABLE W2V EMBEDDING")
    if options["is_bidirectional"]:
        print("USE BI-DIRECTIONAL RNN")
    if options["omit_eos"]:
        print("<eos> IS OMITTED IN TESTING DATA")
    if options["prediction_bytes_limitation"]:
        print("MAXIMUM BYTES IN PREDICTION IS LIMITED")
    print("RNN TYPE: " + options["cell"])
    for k in consts:
        print(k + ":", consts[k])

def init_modules():
    
    init_seeds()

    options = {}

    options["is_debugging"] = False
    options["is_predicting"] = False
    options["model_selection"] = False # When options["is_predicting"] = True, true means use validation set for tuning, false is real testing.

    options["cuda"] = cfg.CUDA and torch.cuda.is_available()
    options["device"] = torch.device("cuda" if  options["cuda"] else "cpu")
    
    #in config.py
    options["cell"] = cfg.CELL
    options["copy"] = cfg.COPY
    options["coverage"] = cfg.COVERAGE
    options["is_bidirectional"] = cfg.BI_RNN
    options["avg_nll"] = cfg.AVG_NLL

    options["beam_decoding"] = cfg.BEAM_SEARCH # False for greedy decoding
    
    assert TRAINING_DATASET_CLS.IS_UNICODE == TESTING_DATASET_CLS.IS_UNICODE
    options["is_unicode"] = TRAINING_DATASET_CLS.IS_UNICODE # True Chinese dataet
    options["has_y"] = TRAINING_DATASET_CLS.HAS_Y
    
    options["has_learnable_w2v"] = True
    options["omit_eos"] = False # omit <eos> and continuously decode until length of sentence reaches MAX_LEN_PREDICT (for DUC testing data)
    options["prediction_bytes_limitation"] = False if TESTING_DATASET_CLS.MAX_BYTE_PREDICT == None else True
    options["fire"] = cfg.FIRE

    assert options["is_unicode"] == False

    consts = {}
    
    consts["idx_gpu"] = cudaid

    consts["norm_clip"] = cfg.NORM_CLIP
    consts["dim_x"] = cfg.DIM_X
    consts["dim_y"] = cfg.DIM_Y
    consts["len_x"] = cfg.MAX_LEN_X + 1 # plus 1 for eos
    consts["len_y"] = cfg.MAX_LEN_Y + 1
    consts["num_x"] = cfg.MAX_NUM_X
    consts["num_y"] = cfg.NUM_Y
    consts["hidden_size"] = cfg.HIDDEN_SIZE
    consts["d_ff"] = cfg.FF_SIZE
    consts["num_heads"] = cfg.NUM_H
    consts["dropout"] = cfg.DROPOUT
    consts["num_layers"] = cfg.NUM_L
    consts["label_smoothing"] = cfg.SMOOTHING
    consts["alpha"] = cfg.ALPHA
    consts["beta"] = cfg.BETA

    consts["batch_size"] = 5 if options["is_debugging"] else TRAINING_DATASET_CLS.BATCH_SIZE
    if options["is_debugging"]:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 2
    else:
        #consts["testing_batch_size"] = 1 if options["beam_decoding"] else TESTING_DATASET_CLS.BATCH_SIZE 
        consts["testing_batch_size"] = TESTING_DATASET_CLS.BATCH_SIZE

    consts["min_len_predict"] = TESTING_DATASET_CLS.MIN_LEN_PREDICT
    consts["max_len_predict"] = TESTING_DATASET_CLS.MAX_LEN_PREDICT
    consts["max_byte_predict"] = TESTING_DATASET_CLS.MAX_BYTE_PREDICT
    consts["testing_print_size"] = TESTING_DATASET_CLS.PRINT_SIZE

    consts["lr"] = cfg.LR
    consts["beam_size"] = cfg.BEAM_SIZE

    consts["max_epoch"] = 50 if options["is_debugging"] else 64 
    consts["print_time"] = 2
    consts["save_epoch"] = 1

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1

    modules = {}
    
    [_, dic, hfw, w2i, i2w, w2w] = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "dic.pkl", "rb")) 
    consts["dict_size"] = len(dic)
    modules["dic"] = dic
    modules["w2i"] = w2i
    modules["i2w"] = i2w
    modules["lfw_emb"] = modules["w2i"][cfg.W_UNK]
    modules["eos_emb"] = modules["w2i"][cfg.W_EOS]
    modules["bos_idx"] = modules["w2i"][cfg.W_BOS]     
    consts["pad_token_idx"] = modules["w2i"][cfg.W_PAD]

    return modules, consts, options

def beam_decode(fname, batch, model, modules, consts, options):
    fname = str(fname)

    beam_size = consts["beam_size"]
    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size)

    last_traces = [[]]
    last_scores = torch.FloatTensor(np.zeros(1)).to(options["device"])
    last_c_scores = torch.FloatTensor(np.zeros(1)).to(options["device"])
    last_states = [[]]

    if options["copy"]:
        x, x_mask, word_emb, padding_mask, y, len_y, ref_sents, max_ext_len, oovs = batch
    else:
        x, word_emb, padding_mask, y, len_y, ref_sents = batch

    ys = torch.LongTensor(np.ones((1, num_live), dtype="int64") * modules["bos_idx"]).to(options["device"])
    x = x.unsqueeze(1)
    word_emb = word_emb.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(1)
    if options["copy"]:
        x_mask = x_mask.unsqueeze(1)

    for step in range(consts["max_len_predict"]):
        tile_word_emb = word_emb.repeat(1, num_live, 1)
        tile_padding_mask = padding_mask.repeat(1, num_live)
        if options["copy"]: 
            tile_x = x.repeat(1, num_live)
            tile_x_mask = x_mask.repeat(1, num_live, 1) 
        
        if options["copy"]:
            y_pred, attn_dist = model.decode(ys, tile_x_mask, None, tile_word_emb, tile_padding_mask, tile_x, max_ext_len)
        else:
            y_pred, attn_dist = model.decode(ys, None, None, tile_word_emb, tile_padding_mask)
        
        dict_size = y_pred.shape[-1]
        y_pred = y_pred[-1, :, :] 
        if options["coverage"]:
            attn_dist = attn_dist[-1, :, :]

        cand_y_scores = last_scores + torch.log(y_pred) # larger is better
        if options["coverage"]:
            cand_scores = (cand_y_scores + last_c_scores).flatten()
        else:
            cand_scores = cand_y_scores.flatten()
        idx_top_joint_scores = torch.topk(cand_scores, beam_size - num_dead)[1]

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_y_scores.flatten()[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead))
        states_now = []    

        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            if options["coverage"]:
                states_now.append(last_states[j] + [copy.copy(attn_dist[j, :])])
            
        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []
        last_c_scores = []
        dead_ids = []
        for i in range(len(traces_now)):
            if traces_now[i][-1] == modules["eos_emb"] and len(traces_now[i]) >= consts["min_len_predict"]:
                samples.append([str(e.item()) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
                dead_ids += [i] 
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
            
                if options["coverage"]:
                    last_states.append(states_now[i])  
                    attns = torch.stack(states_now[i])
                    m, n = attns.shape
                    cp = torch.sum(attns, dim=0)
                    cp = torch.max(cp, torch.ones_like(cp))
                    cp = - consts["beta"] * (torch.sum(cp).item() - n)
                    last_c_scores.append(cp)            

                num_live += 1
        if num_live == 0 or num_dead >= beam_size:
            break
    
        if options["coverage"]:
            last_c_scores = torch.FloatTensor(np.array(last_c_scores).reshape((num_live, 1))).to(options["device"])
        
        last_scores = torch.FloatTensor(np.array(last_scores).reshape((num_live, 1))).to(options["device"])
        next_y = []
        for e in last_traces:
            eid = e[-1].item()
            if eid in modules["i2w"]:
                next_y.append(eid)
            else:
                next_y.append(modules["lfw_emb"]) # unk for copy mechanism

        next_y = np.array(next_y).reshape((1, num_live))
        next_y = torch.LongTensor(next_y).to(options["device"])
        
        if step == 0:
            ys = ys.repeat(1, num_live)
        ys_ = []
        py_ = []
        for i in range(ys.size(1)):
            if i not in dead_ids:
                ys_.append(ys[:, i])
        ys = torch.cat([torch.stack(ys_, dim=1), next_y], dim=0)

        assert num_live + num_dead == beam_size

    if num_live > 0:
        for i in range(num_live):
            samples.append([str(e.item()) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1
    
    #weight by length
    for i in range(len(sample_scores)):
        sent_len = float(len(samples[i]))
        lp = np.power(5 + sent_len, consts["alpha"]) / np.power(5 + 1, consts["alpha"])
        sample_scores[i] /= lp
    
    idx_sorted_scores = np.argsort(sample_scores) # ascending order
    if options["has_y"]:
        ly = len_y[0]
        y_true = y[0 : ly].tolist()
        y_true = [str(i) for i in y_true[:-1]] # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) >= consts["min_len_predict"]:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    num_samples = len(sorted_samples)
    if len(sorted_samples) == 1:
        sorted_samples = sorted_samples[0]
        num_samples = 1

    # for task with bytes-length limitation 
    if options["prediction_bytes_limitation"]:
        for i in range(len(sorted_samples)):
            sample = sorted_samples[i]
            b = 0
            for j in range(len(sample)):
                e = int(sample[j]) 
                if e in modules["i2w"]:
                    word = modules["i2w"][e]
                else:
                    word = oovs[e - len(modules["i2w"])]
                if j == 0:
                    b += len(word)
                else:
                    b += len(word) + 1 
                if b > consts["max_byte_predict"]:
                    sorted_samples[i] = sorted_samples[i][0 : j]
                    break

    dec_words = []

    for e in sorted_samples[-1]:
        e = int(e)
        if e in modules["i2w"]: # if not copy, the word are all in dict
            dec_words.append(modules["i2w"][e])
        else:
            dec_words.append(oovs[e - len(modules["i2w"])])
    
    write_for_rouge(fname, ref_sents, dec_words, cfg)

    # beam search history for checking
    if not options["copy"]:
        oovs = None
    write_summ("".join((cfg.cc.BEAM_SUMM_PATH, fname)), sorted_samples, num_samples, options, modules["i2w"], oovs, sorted_scores)
    write_summ("".join((cfg.cc.BEAM_GT_PATH, fname)), y_true, 1, options, modules["i2w"], oovs) 



def predict(model, modules, consts, options):
    print("start predicting,")
    model.eval()
    options["has_y"] = TESTING_DATASET_CLS.HAS_Y
    if options["beam_decoding"]:
        print("using beam search")
    else:
        print("using greedy search")
    rebuild_dir(cfg.cc.BEAM_SUMM_PATH)
    rebuild_dir(cfg.cc.BEAM_GT_PATH)
    rebuild_dir(cfg.cc.GROUND_TRUTH_PATH)
    rebuild_dir(cfg.cc.SUMM_PATH)

    print("loading test set...")
    if options["model_selection"]:
        xy_list = pickle.load(open(cfg.cc.VALIDATE_DATA_PATH + "pj1000.pkl", "rb")) 
    else:
        xy_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test.pkl", "rb")) 
    batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)

    print("num_files = ", num_files, ", num_batches = ", num_batches)
    
    running_start = time.time()
    partial_num = 0
    total_num = 0
    si = 0
    for idx_batch in range(num_batches):
        test_idx = batch_list[idx_batch]
        batch_raw = [xy_list[xy_idx] for xy_idx in test_idx]
        batch = datar.get_data(batch_raw, modules, consts, options)
        
        assert len(test_idx) == batch.x.shape[1] # local_batch_size

                    
        word_emb, padding_mask = model.encode(torch.LongTensor(batch.x).to(options["device"]))

        if options["beam_decoding"]:
            for idx_s in range(len(test_idx)):
                if options["copy"]:
                    inputx = (torch.LongTensor(batch.x_ext[:, idx_s]).to(options["device"]), \
                            torch.FloatTensor(batch.x_mask[:, idx_s, :]).to(options["device"]), \
                          word_emb[:, idx_s, :], padding_mask[:, idx_s],\
                          batch.y[:, idx_s], [batch.len_y[idx_s]], batch.original_summarys[idx_s],\
                          batch.max_ext_len, batch.x_ext_words[idx_s])
                else:
                    inputx = (torch.LongTensor(batch.x[:, idx_s]).to(options["device"]), word_emb[:, idx_s, :], padding_mask[:, idx_s],\
                              batch.y[:, idx_s], [batch.len_y[idx_s]], batch.original_summarys[idx_s])

                beam_decode(si, inputx, model, modules, consts, options)
                si += 1
        else:
            pass
            #greedy_decode()

        testing_batch_size = len(test_idx)
        partial_num += testing_batch_size
        total_num += testing_batch_size
        if partial_num >= consts["testing_print_size"]:
            print(total_num, "summs are generated")
            partial_num = 0
    print (si, total_num)

def run(existing_model_name = None):
    modules, consts, options = init_modules()

    if options["is_predicting"]:
        need_load_model = True
        training_model = False
        predict_model = True
    else:
        need_load_model = False
        training_model = True
        predict_model = False

    print_basic_info(modules, consts, options)

    if training_model:
        print ("loading train set...")
        if options["is_debugging"]:
            xy_list = pickle.load(open(cfg.cc.TESTING_DATA_PATH + "test.pkl", "rb")) 
        else:
            xy_list = pickle.load(open(cfg.cc.TRAINING_DATA_PATH + "train.pkl", "rb")) 
        batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
        print ("num_files = ", num_files, ", num_batches = ", num_batches)

    running_start = time.time()
    if True: #TODO: refactor
        print ("compiling model ..." )
        model = Model(modules, consts, options)
        if options["cuda"]:
            model.cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=consts["lr"], initial_accumulator_value=0.1)
        
        model_name = "".join(["cnndm.s2s.", options["cell"]])
        existing_epoch = 0
        if need_load_model:
            if existing_model_name == None:
                existing_model_name = "cnndm.s2s.transformer.gpu0.epoch27.2"
            print ("loading existed model:", existing_model_name)
            model, optimizer = load_model(cfg.cc.MODEL_PATH + existing_model_name, model, optimizer)

        if training_model:
            print ("start training model ")
            model.train()
            print_size = num_files // consts["print_time"] if num_files >= consts["print_time"] else num_files

            last_total_error = float("inf")
            print ("max epoch:", consts["max_epoch"])
            for epoch in range(0, consts["max_epoch"]):
                print ("epoch: ", epoch + existing_epoch)
                num_partial = 1
                total_error = 0.0
                error_c = 0.0
                partial_num_files = 0
                epoch_start = time.time()
                partial_start = time.time()
                # shuffle the trainset
                batch_list, num_files, num_batches = datar.batched(len(xy_list), options, consts)
                used_batch = 0.
                for idx_batch in range(num_batches):
                    train_idx = batch_list[idx_batch]
                    batch_raw = [xy_list[xy_idx] for xy_idx in train_idx]
                    if len(batch_raw) != consts["batch_size"]:
                        continue
                    local_batch_size = len(batch_raw)
                    batch = datar.get_data(batch_raw, modules, consts, options)
                  
                    
                    model.zero_grad()
                    
                    y_pred, cost = model(torch.LongTensor(batch.x).to(options["device"]),\
                                   torch.LongTensor(batch.y_inp).to(options["device"]),\
                                   torch.LongTensor(batch.y).to(options["device"]),\
                                   torch.FloatTensor(batch.x_mask).to(options["device"]),\
                                   torch.FloatTensor(batch.y_mask).to(options["device"]),\
                                   torch.LongTensor(batch.x_ext).to(options["device"]),\
                                   torch.LongTensor(batch.y_ext).to(options["device"]),\
                                   batch.max_ext_len)


                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), consts["norm_clip"])
                    optimizer.step()

                    
                    cost = cost.item()
                    total_error += cost
                    used_batch += 1
                    partial_num_files += consts["batch_size"]
                    if partial_num_files // print_size == 1 and idx_batch < num_batches:
                        print (idx_batch + 1, "/" , num_batches, "batches have been processed,", \
                                "average cost until now:", "cost =", total_error / used_batch, ",", \
                                "cost_c =", error_c / used_batch, ",", \
                                "time:", time.time() - partial_start)
                        partial_num_files = 0
                        if not options["is_debugging"]:
                            print("save model... ",)
                            file_name =  model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial)
                            save_model(cfg.cc.MODEL_PATH + file_name, model, optimizer)
                            if options["fire"]:
                                shutil.move(cfg.cc.MODEL_PATH + file_name, "/out/")

                            print("finished")
                        num_partial += 1
                print ("in this epoch, total average cost =", total_error / used_batch, ",", \
                        "cost_c =", error_c / used_batch, ",",\
                        "time:", time.time() - epoch_start)

                print_sent_dec(y_pred, batch.y, batch.y_mask, batch.x_ext_words, modules, consts, options, local_batch_size)
                
                if last_total_error > total_error or options["is_debugging"]:
                    last_total_error = total_error
                    if not options["is_debugging"]:
                        print ("save model... ",)
                        file_name =  model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial)
                        save_model(cfg.cc.MODEL_PATH + file_name, model, optimizer)
                        if options["fire"]:
                            shutil.move(cfg.cc.MODEL_PATH + file_name, "/out/")

                        print ("finished")
                else:
                    print ("optimization finished")
                    break

            print ("save final model... "),
            file_name = model_name + ".final.gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch // consts["save_epoch"] + existing_epoch) + "." + str(num_partial)
            save_model(cfg.cc.MODEL_PATH + file_name, model, optimizer)
            if options["fire"]:
                shutil.move(cfg.cc.MODEL_PATH + file_name, "/out/")

            print ("finished")
        else:
            print ("skip training model")

        if predict_model:
            predict(model, modules, consts, options)
    print ("Finished, time:", time.time() - running_start)

if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    existing_model_name = sys.argv[1] if len(sys.argv) > 1 else None
    run(existing_model_name)
