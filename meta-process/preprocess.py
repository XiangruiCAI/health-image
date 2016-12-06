#! /usr/bin/python
# encoding=utf-8

import csv
import os
import glob
import dicom


IMAGEMALE = '../data/raw/DX_male/'
IMAGEFEMALE = '../data/raw/DX_female/'


LOG = open('preprocess.log', 'w')
NOPA = open('nopa.log', 'w')


def find_postero(prefix, img_list):
    '''
    find the positive images in a list
    return index if found -1 otherwise
    '''
    flag = False
    for i in range(len(img_list)):
        if '.dcm' in img_list[i].lower():
            img_path = os.path.join(prefix, img_list[i])
            # print img_path
            ref = dicom.read_file(img_path)
            if hasattr(ref, 'ViewPosition') and ref.ViewPosition == 'PA':
                flag = True
            if hasattr(ref, 'AcquisitionDeviceProcessingDescription') \
                    and 'postero' in ref.AcquisitionDeviceProcessingDescription:
                flag = True
        if flag == True:
            return i
        i += 1

    if flag == False:
        return -1


global num_missing
num_missing = 0
global num_nopa
num_nopa = 0

def find_record(name):
    '''
    find the image path in the two folder
    name: the given folder name
    return image path and '' if no images found
    '''

    # print 'in find_record, name: ' + name
    prefix = ''
    m_cand = glob.glob(IMAGEMALE + name + '*')
    f_cand = glob.glob(IMAGEFEMALE + name + '*')
    if len(m_cand) > 1:
        LOG.write('More than 1 candidate in male folder: ' + name + '\n')
        return ''
    elif len(m_cand) == 1:
        prefix = m_cand[0]
    else:
        if len(f_cand) > 1:
            LOG.write('More than 1 candidate in female folder: ' + name + '\n')
            return ''
        elif len(f_cand) == 1:
            prefix = f_cand[0]
        else:
            LOG.write('Image folder not found: ' + name + '\n')
            global num_missing
            num_missing += 1
            return ''

    images = os.listdir(prefix)
    if len(images) == 0:
        LOG.write('Image folder is empty: ' + prefix[12:] + '\n')
        return ''

    idx = 0
    if len(images) > 1:
        idx = find_postero(prefix, images)

    if idx == -1:
        LOG.write('Postero image not found: ' + prefix[12:] + '\n')
        NOPA.write(prefix[12:] + '\n')
        global num_nopa
        num_nopa += 1
        return ''

    path = os.path.join(prefix, images[idx])
    return path


global num_total
num_total = 0

def meta_data(iline):
    '''
    generate a line of the meta data for trainning
    list: a input line in the records.csv file
    return: a list containing path, label, IID, gender
    '''

    meta = []
    iid = iline[0].strip()
    uid = iline[1].strip()
    report_desc = iline[2].strip().strip('\"').replace('、', '')
    study_desc = iline[3].strip().strip('\"')

    label = 1
    gender = 1
    if study_desc == '胸部正位片(DR)' or study_desc == '胸部正侧位片(DR)':
        global num_total
        num_total += 1
        # print 'hello, in study_desc'
        path = find_record(uid)
        # print 'path: ' + path
        if path == '':
            return meta

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
                return meta
            label = 0

        if IMAGEMALE in path:
            gender = 1
        else:
            gender = 0

        meta.append(path[12:])
        meta.append(label)
        meta.append(iid)
        meta.append(gender)

    return meta


if __name__ == '__main__':
    with open('../data/raw/records.csv', 'rb') as csvinput, \
            open('./meta.csv', 'w') as csvoutput1,  \
            open('./meta_full.csv', 'w') as csvoutput2:
        RECORDS = csv.reader(csvinput)
        next(RECORDS, None)
        HEADER = ['path', 'label', 'IID', 'gender']

        OUT1 = csv.writer(csvoutput1, delimiter=' ')
        OUT2 = csv.writer(csvoutput2, delimiter=' ')
        OUT1.writerow(HEADER[:2])
        OUT2.writerow(HEADER)

        num_male = 0
        num_female = 0
        num_pos = 0
        num_neg = 0
        #count = 0
        for line in RECORDS:
            # print line
            metaline = meta_data(line)
            if metaline:
                OUT1.writerow(metaline[:2])
                OUT2.writerow(metaline)
                if metaline[1] == 1:
                    num_pos += 1
                else:
                    num_neg += 1
                if metaline[-1] == 1:
                    num_male += 1
                else:
                    num_female += 1
            #count += 1
            # if count == 10: break
        if num_male + num_female != num_pos + num_neg:
            print "Error!"
        else:
            #global num_total
            print "==================================================="
            print "number of total samples in records.csv: ", num_total
            print "number of missing folder: ", num_missing
            print "number of no PA: ", num_nopa
            print "---------------------------------------------------"
            print "number of male samples: ", num_male
            print "number of female samples: ", num_female
            print "number of positive samples: ", num_pos
            print "number of negative samples: ", num_neg
            print "==================================================="

            LOG.write("===================================================\n")
            LOG.write("number of total samples in records.csv: %d\n" %num_total)
            LOG.write("number of missing folder: %d\n" %num_missing)
            LOG.write("number of no PA: %d\n" %num_nopa)
            LOG.write("---------------------------------------------------\n")
            LOG.write("number of male samples: %d\n" %num_male)
            LOG.write("number of female samples: %d\n" %num_female)
            LOG.write("number of positive samples: %d\n" %num_pos)
            LOG.write("number of negative samples: %d\n" %num_neg)
            LOG.write("===================================================\n")

    LOG.close()
