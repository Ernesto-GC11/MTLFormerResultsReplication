"""
Python 3 / tf.keras port of evaluate_next_activity_and_time.py.
Changes: arguments instead of hard-coded names, results go to <result_dir>/<dataset>/seed<N>/, model(...) instead of model.predict(...), and times2/times3 are reset per case (upstream shares one list between all cases, which gives every test case the timestamps of the first case of the log; use --original_time_bug to keep that).
Run from the 'code/' folder: python evaluate_next_activity_and_time.py --dataset helpdesk --seed 1
"""
from __future__ import division
import argparse
import glob
import os
import re

parser = argparse.ArgumentParser(description="Evaluate a trained model.")

parser.add_argument("--dataset", required=True,
                    help="csv name in ../data without extension: helpdesk, bpi_12_w, bpi_12_w_no_repeat, env_permit")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed (a random one is drawn and printed if omitted)")
parser.add_argument("--data_dir", default="../data")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--model_dir", default="output_files/models")
parser.add_argument("--result_dir", default="output_files/results")
parser.add_argument("--model_path", default=None,
                    help="explicit .h5 file (default: best checkpoint in <model_dir>/<dataset>/seed<N>)")
parser.add_argument("--max_cases", type=int, default=None,
                    help="only evaluate the first N test cases (smoke tests only)")
parser.add_argument("--original_time_bug", action="store_true",
                    help="reproduce upstream behaviour where times2/times3 lists are shared between cases")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import csv
import copy
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import distance
from jellyfish import damerau_levenshtein_distance
from sklearn import metrics
from tensorflow.keras.models import load_model

if args.seed is None and args.model_path is None:
    parser.error("--seed (or --model_path) is required to locate the trained model")
eventlog = args.dataset + ".csv"
run_name = "seed%s" % args.seed if args.seed is not None else "custom"
result_dir = os.path.join(args.result_dir, args.dataset, run_name)
os.makedirs(result_dir, exist_ok=True)

csvfile = open(os.path.join(args.data_dir, eventlog), 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []
timeseqs2 = []
times = []
times2 = []
numlines = 0
casestarttime = None
lasteventtime = None
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
        line = ''
        times = []
        if not args.original_time_bug:
            times2 = []
            times3 = []
        numlines+=1
    line+=chr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_c = caseids[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_c = caseids[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]

lines = fold1 + fold2
caseids = fold1_c + fold2_c
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

step = 1
sentences = []
softness = 0
next_chars = []
lines = list(map(lambda x: x+'!',lines))
maxlen = max(map(lambda x: len(x),lines))

chars = list(map(lambda x : set(x),lines))
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
print(indices_char)


lastcase = ''
line = ''
firstLine = True
lines = []
caseids = []
timeseqs = []  # relative time since previous event
timeseqs2 = [] # relative time since case start
timeseqs3 = [] # absolute time of previous event
times = []
times2 = []
times3 = []
numlines = 0
casestarttime = None
lasteventtime = None
csvfile = open(os.path.join(args.data_dir, eventlog), 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        caseids.append(row[0])
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
        line = ''
        times = []
        if not args.original_time_bug:
            times2 = []
            times3 = []
        numlines+=1
    line+=chr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    #timediff = log(timediff+1)
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
numlines+=1

fold3 = lines[2*elems_per_fold:]
fold3_c = caseids[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
#fold3_t4 = timeseqs4[2*elems_per_fold:]

lines = fold3
caseids = fold3_c
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
if args.max_cases:
    lines, caseids, lines_t, lines_t2, lines_t3 = [x[:args.max_cases] for x in (lines, caseids, lines_t, lines_t2, lines_t3)]
#lines_t4 = fold1_t4 + fold2_t4

# set parameters
predict_size = 1

# load model, set this to the model generated by train.py
model_path = args.model_path
if model_path is None:
    ckpts = glob.glob(os.path.join(args.model_dir, args.dataset, 'seed%d' % args.seed, 'model_*.h5'))
    assert ckpts, 'no checkpoint found - run train.py first'
    # save_best_only=True: the checkpoint with the highest epoch number is the best one
    model_path = max(ckpts, key=lambda p: int(re.search(r'model_(\d+)-', os.path.basename(p)).group(1)))
print('loading', model_path)
model = load_model(model_path, compile=False)

# define helper functions
def encode(sentence, times, times3, maxlen=maxlen):
    num_features = len(chars)+5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
    return X

def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []


# make predictions
with open(os.path.join(result_dir, 'next_activity_and_time_%s' % eventlog), 'w', newline='', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["CaseID", "Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE"])
    for prefix_size in range(2,maxlen):
        print(prefix_size)
        for line, caseid, times, times3 in zip(lines, caseids, lines_t, lines_t3):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            if '!' in cropped_line:
                continue # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
            ground_truth_t = times[prefix_size:prefix_size+predict_size]
            predicted = ''
            predicted_t = []
            for i in range(predict_size):
                if len(ground_truth)<=i:
                    continue
                enc = encode(cropped_line, cropped_times, cropped_times3)
                y = model(enc, training=False)
                y_char = y[0][0].numpy()
                y_t = float(y[1][0][0])
                prediction = getSymbol(y_char)              
                cropped_line += prediction
                if y_t<0:
                    y_t=0
                cropped_times.append(y_t)
                y_t = y_t * divisor
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                predicted_t.append(y_t)
                if i==0:
                    if len(ground_truth_t)>0:
                        one_ahead_pred.append(y_t)
                        one_ahead_gt.append(ground_truth_t[0])
                if i==1:
                    if len(ground_truth_t)>1:
                        two_ahead_pred.append(y_t)
                        two_ahead_gt.append(ground_truth_t[1])
                if i==2:
                    if len(ground_truth_t)>2:
                        three_ahead_pred.append(y_t)
                        three_ahead_gt.append(ground_truth_t[2])
                if prediction == '!': # end of case was just predicted, therefore, stop predicting further into the future
                    print('! predicted, end case')
                    break
                predicted += prediction
            output = []
            if len(ground_truth)>0:
                output.append(caseid)
                output.append(prefix_size)
                output.append(ground_truth)
                output.append(predicted)
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(len(predicted),len(ground_truth)))
                if dls<0:
                    dls=0 # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                output.append(dls)
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append('; '.join(str(x) for x in ground_truth_t))
                output.append('; '.join(str(x) for x in predicted_t))
                if len(predicted_t)>len(ground_truth_t): # if predicted more events than length of case, only use needed number of events for time evaluation
                    predicted_t = predicted_t[:len(ground_truth_t)]
                if len(ground_truth_t)>len(predicted_t): # if predicted less events than length of case, put 0 as placeholder prediction
                    predicted_t.extend(range(len(ground_truth_t)-len(predicted_t)))
                if len(ground_truth_t)>0 and len(predicted_t)>0:
                    output.append('')
                    output.append(metrics.mean_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                    #output.append(metrics.median_absolute_error([ground_truth_t[0]], [predicted_t[0]]))
                else:
                    output.append('')
                    output.append('')
                    output.append('')
                spamwriter.writerow(output)