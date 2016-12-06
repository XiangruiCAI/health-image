#! /usr/bin/python
# encoding=utf-8

import csv
import os
import random


LOG = open('combine.log', 'w')

def find_label(path):
    '''
    find label by path for manual file
    '''
    gender = 1
    if 'female' in path:
        gender = 0
    else:
        gender = 1
    with open('../data/raw/records.csv', 'rb') as r:
        label = -1
        iid = ''
        RECORDS = csv.reader(r)
        next(RECORDS, None)
        for item in RECORDS:
            iid = item[0].strip()
            uid = item[1].strip()
            report_desc = item[2].strip().strip('\"').replace('、', '')
            if uid in path:
                if report_desc == '心肺膈未见明显异常。' \
                    or report_desc == '心肺膈未见明确异常。' \
                    or report_desc == '心肺膈未见明显病变。' \
                    or report_desc == '心肺膈未见明确病变。' \
                    or report_desc == '双肺、心、膈未见明确异常。' \
                    or report_desc == '双肺、心、膈未见明显异常。' \
                    or report_desc == '双肺、心膈未见明确异常。' \
                    or report_desc == '双肺未见明确病变。' \
                    or report_desc == '双肺、心、膈未见异常。' \
                    or report_desc == '心、肺、膈未见明显异常。' \
                    or report_desc == '心、肺、双膈未见明显异常。' \
                    or report_desc == '心肺膈未见明确异常，请结合临床，必要时CT进一步观察。' \
                    or report_desc == '心肺膈未见明确异常，请结合临床查体，必要时进一步检查。' \
                    or report_desc == '胸正位片未见明显异常。':
                    label = 1
                else:
                    if '心肺膈未见明显异常' in report_desc and 'PICC' in report_desc:
                        LOG.write('Fuzzy case(心肺膈未见明显异常 and PICC管): ' + uid + '\n')
                        label = -2
                        break
                    label = 0
                break
        if label == -1:
            LOG.write('Label not found: ' + path + '\n')
        return label, iid, gender


global num_dupl
num_dupl = 0
global num_confl
num_confl = 0


def remove_duplicate(meta_path):
    meta = {}
    with open(meta_path, 'rb') as a, open('./manually.csv') as m:
        auto = csv.reader(a, delimiter = ' ')
        #i = 0
        #for item in auto:
        #    i += 1
        #print "#auto line: ", i
        next(auto, None) #skip header
        for item in auto:
            if item[0] in meta:
                if meta[item[0]][1] != item[1]:
                    global num_confl
                    num_confl += 1
                    LOG.write('Conflict case: ' + item[0] + '\n')
                else:
                    global num_dupl
                    num_dupl += 1
                    LOG.write('duplicate case: ' + item[0] + '\n')
            else:
                meta[item[0]] = item

        print
        manual = csv.reader(m, delimiter = '\t')
        #i = 0
        #for item in manual:
        #    i += 1
        #print '#manual line: ', i
        for item in manual:
            l = []
            path = os.path.join(item[0], item[1])
            lbl, iid, gd = find_label(path)
            if lbl >= 0:
                l.append(path)
                l.append(lbl)
                if 'full' in meta_path:
                    l.append(iid)
                    l.append(gd)
                if l[0] in meta:
                    if meta[l[0]][1] != l[1]:
                        num_confl += 1
                        LOG.write('Conflict case: ' + l[0] + '\n')
                    else:
                        num_dupl += 1
                        LOG.write('duplicate case: ' + item[0] + '\n')
                else:
                    meta[l[0]] = l
    return meta


def split_write(l, ratio, w1, w2):
    random.shuffle(l)
    spoint = int(len(l) * ratio)
    for i in range(len(l)):
        if i < spoint:
            w1.writerow(l[i])
        else:
            w2.writerow(l[i])
    LOG.write('number of training images: %d\n' %spoint)
    LOG.write('number of test images: %d\n' %(len(l)-spoint))
    return spoint


if __name__ == '__main__':
    with open('./lut.log', 'rb') as lut, \
            open('./meta_full_final.csv', 'w') as o1, \
            open('./meta_final.csv', 'w') as o2, \
            open('./meta_train.csv', 'w') as o3, \
            open('./meta_test.csv', 'w') as o4:
        no_lut = {}
        i = 0
        for line in lut:
            no_lut[line.strip('\n')] = i
            i += 1

        meta = []
        header = ['path', 'label', 'IID', 'gender']
        out1 = csv.writer(o1, delimiter = ' ')
        out2 = csv.writer(o2, delimiter = ' ')
        train = csv.writer(o3, delimiter = ' ')
        test = csv.writer(o4, delimiter = ' ')
        out1.writerow(header)
        out2.writerow(header[:2])
        meta_dict = remove_duplicate('./meta_full.csv')
        num_pos = 0
        num_neg = 0
        num_male = 0
        num_female = 0
        for key in sorted(meta_dict.keys()):
            if key in no_lut:
                continue
            out1.writerow(meta_dict[key])
            out2.writerow(meta_dict[key][:2])
            meta.append(meta_dict[key][:2])
            if int(meta_dict[key][1]) == 1:
                num_pos += 1
            else:
                num_neg += 1
            if int(meta_dict[key][-1]) == 1:
                num_male += 1
            else:
                num_female += 1

        if num_male + num_female != num_pos + num_neg:
            print "Error!"
        else:
            #global num_total
            print "==================================================="
            print "number of total samples finally: ", len(meta_dict)
            print "number of confict samples finally: ", num_confl
            print "number of duplicate samples finally: ", num_dupl
            print "---------------------------------------------------"
            print "number of male samples: ", num_male
            print "number of female samples: ", num_female
            print "number of positive samples: ", num_pos
            print "number of negative samples: ", num_neg
            print "==================================================="

            LOG.write("===================================================\n")
            LOG.write("number of total samples finally: %d\n" %len(meta_dict))
            LOG.write("number of confict samples finally: %d\n" %num_confl)
            LOG.write("number of duplicate samples finally: %d\n" %num_dupl)
            LOG.write("---------------------------------------------------\n")
            LOG.write("number of male samples: %d\n" %num_male)
            LOG.write("number of female samples: %d\n" %num_female)
            LOG.write("number of positive samples: %d\n" %num_pos)
            LOG.write("number of negative samples: %d\n" %num_neg)
            LOG.write("===================================================\n")
        s = split_write(meta, 0.9, train, test)
        print ('number of training images: %d' %s)
        print ('number of test images: %d' %(len(meta)-s))

    LOG.close()
