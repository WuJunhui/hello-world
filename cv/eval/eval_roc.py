import numpy
from sklearn import metrics

labelmap={'porn':0,'sexy':1, 'normal':2}
labelnum=3


def check_EER_and_Accuracy(srcf, test_label):
    accuracy = 0.0
    y_tests = [] # label like [0, 0,1]
    y_preds = [] # three scores

    with open(srcf,'r') as f:
        lines = f.readlines()

    for line in lines:
        #label=labelmap[line.split('/')[0]]
        label=line.split(' ')[1]
        label=int(label)
        y_test = [0] * labelnum
        y_test[label] = 1
        y_tests.append(y_test)

        y_pred=line.rstrip().split(' ')[-labelnum:]
        y_preds.append(y_pred)

        if (numpy.array(y_pred,dtype=float)).argmax()==label:
            accuracy += 1

    accuracy /= len(lines)


    y_tests = numpy.array(y_tests)
    y_preds = numpy.array(y_preds,dtype=float)

    #print 'y_tests',y_tests.shape
    #print 'y_preds',y_preds.shape

    fpr, tpr, thres = metrics.roc_curve(y_tests[:, test_label], y_preds[:, test_label])
    auc = metrics.auc(fpr, tpr)

    min_val = 100
    final_sub = 0

    for i in range(len(fpr)):
        if abs(fpr[i] + tpr[i] - 1.0) < min_val:
            min_val = abs(fpr[i] + tpr[i] - 1.0)
            final_sub = i

    thres_final = thres[final_sub]

    cnt_wrong_loujian, cnt_all_loujian, cnt_wrong_wujian, cnt_all_wujian = 0, 0, 0, 0

    for i in range(0, len(y_tests)):
        label = numpy.argmax(y_tests[i])
        score = y_preds[i][test_label]

        if label == test_label:
            cnt_all_loujian += 1
            if score < thres_final:
                cnt_wrong_loujian += 1
        else:
            cnt_all_wujian += 1
            if score >= thres_final:
                cnt_wrong_wujian += 1

    output_line = '{} {} {}/{} {}/{} {}'.format(thres_final, auc, cnt_wrong_loujian, cnt_all_loujian, cnt_wrong_wujian, cnt_all_wujian, accuracy)

    return output_line

if __name__=="__main__":
    outline = check_EER_and_Accuracy('out.txt', 0)
    print outline
