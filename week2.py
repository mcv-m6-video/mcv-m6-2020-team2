import cv2

def task1():
    #TODO Gaussian. Implementation
    return

def task2():
    #TODO Adaptive modeling
    return

def task3():
    '''
    Comparison with the state of the art
    '''

    method='MOG2'
    history=10

    if method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif method == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif method == 'GMG':
        backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
    elif method == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN()
    elif method == 'GSOC':
        backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif method == 'CNT':
        backSub = cv2.bgsegm.createBackgroundSubtractorCNT()
    else:
        raise ValueError(f"Unknown background estimation method {method}. Options are [MOG2, LSBP, GMG, KNN, GSOC, CNT]")

    cap = cv2.VideoCapture('data/AICity_data/train/S03/c010/vdo.avi')
    while cap.isOpened():
        retVal, frame = cap.read()

        fgmask = backSub.apply(frame, learningRate=1.0 / history)

        cv2.imshow('Foreground', cv2.resize(fgmask, (960, 540)))
        cv2.imshow('Original', cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == 27:
            break

def task4():
    #TODO Color sequences
    return

if __name__ == '__main__':
    task3()