
gts_sample=[
    [0,100,100,200,200],
    [3,100,100,200,200],
    [4,100,100,200,200],
    [1,99,99,300,300]
    ]

dets_sample=[
    [0,300,300,400,400],
    [0,99,99,210,210],
    [1,99,99,250,250],
    [1,99,99,280,280],
    [5,99,99,280,280]
    ]

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # bboxA[xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0.0,(xB - xA + 1)) * max(0.0,(yB - yA + 1))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def eval_prec(dets,gts,iou_thresh=0.5):
    tp = 0
    for det in dets:
        for gt in gts:
            if det[0]==gt[0]:
                iou = bb_iou(det[1:],gt[1:])
                print iou
                if iou >= iou_thresh: 
                    tp += 1
    num_dets=len(dets)
    prec=tp/float(num_dets)
    return prec, tp, num_dets

def eval_recall(dets, gts, iou_thresh=0.5):
    hit = 0
    for gt in gts:
        for det in dets:
            if gt[0]==det[0]:
                iou = bb_iou(det[1:],gt[1:])
                print iou
                if iou >= iou_thresh: 
                    hit += 1
                    break
    num_gts=len(gts)
    recall=hit/float(num_gts)
    return recall, hit, num_gts

print bb_iou(dets_sample[0][1:],gts_sample[0][1:])
#print 'prec:', eval_prec(dets_sample,gts_sample,0.5)
#print 'recall:', eval_recall(dets_sample,gts_sample,0.5)