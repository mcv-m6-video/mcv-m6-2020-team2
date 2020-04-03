import argparse
import os
from glob import glob

import numpy as np
from matplotlib import pyplot as plt

import week1, week2, week3, week4
import week5

parser = argparse.ArgumentParser(description='M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring')

parser.add_argument('-w', '--week', type=int, help='week to execute. Options are [1,2,3,4,5]')
parser.add_argument('-t', '--task', type=int, help='task to execute. Options are [1,2,3,4]')
args = parser.parse_args()

path_plots = 'results/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

if args.week == 1:
    if args.task == 1_1:
        week1.task1_1(path_plots)
    elif args.task == 1_2:
        week1.task1_2()
    elif args.task == 2:
        week1.task2(path_plots)
    elif args.task == 3 or args.task == 4:
        week1.task3_4(path_plots)
    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1,2,3,4]")

elif args.week == 2:
    if args.task == 1:
        week2.task1(path_plots)
    elif args.task == 2:
        week2.task2(path_plots)
    elif args.task == 3:
        week2.task3(path_plots)
    elif args.task == 4:
        week2.task4(adaptive=True, random_search=False, color_space='yuv', channels=(1, 2), save_path=None, debug=0)

    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1,2,3,4]")

elif args.week == 3:
    if args.task == 1_1:
        week3.task1_1(architecture='maskrcnn', start=0, length=None, gpu=3, visualize=False)
    elif args.task == 1_2:
        week3.task1_2()
    elif args.task == 2_1:
        week3.task2_1(save_path=None, debug=0)
    elif args.task == 2_2:
        week3.task2_2(debug=0)
    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1_1, 1_2, 2_1, 2_2]")

elif args.week == 4:
    if args.task == 1_1:
        week4.task1_1()
    elif args.task == 1_2:
        week4.task1_2()
    elif args.task == 2_1:
        week4.task2_1()
    elif args.task == 2_2:
        week4.task2_2()
    elif args.task == 3_1:
        week4.task3_1()
    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1_1, 1_2, 2_1, 2_2, 3_1]")

elif args.week == 5:
    if args.task == 1:

        # test_type = 'distance_thresholds'
        # test_type = 'min_width_length'
        test_type = 'one_detector_all_cameras'
        # test_type = 'mean_idf1_across_cameras_sequence_03'

        # TEST DISTANCE THRESHOLDS ON DETECTORS
        if test_type == 'distance_thresholds':

            save_path = 'results/week5/task_1'
            distance_thresholds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
            min_track_len = 5
            min_width = 0
            min_height = 0
            sequence = 'S03'
            camera = 'c010'
            detectors = ['mask_rcnn', 'ssd512', 'yolo3']

            all_idf1s = []
            for detector in detectors:
                idf1s = week5.task1(
                    save_path=save_path,
                    distance_thresholds=distance_thresholds,
                    min_track_len=min_track_len,
                    min_width=min_width,
                    min_height=min_height,
                    sequence=sequence,
                    camera=camera,
                    detector=detector)

                all_idf1s.append(idf1s)

            for idf1s in all_idf1s:
                plt.plot(distance_thresholds, idf1s)
            plt.xticks([d for d in distance_thresholds if d % 100 == 0])
            plt.xlabel('Distance thresholds')
            plt.ylabel('IDF1')
            plt.legend(detectors, loc='best')
            if save_path:
                plt.savefig(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_'+ 'dist-th_vs_idf1.png'))
            plt.show()

        elif test_type == 'min_width_length':

            save_path = 'results/week5/task_1'
            distance_thresholds = [550]
            min_track_len = 5
            min_widths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
            min_heights = [int(0.8*w) for w in min_widths]
            sequence = 'S03'
            camera = 'c010'
            detectors = ['mask_rcnn', 'ssd512', 'yolo3']

            all_idf1s = []
            for detector in detectors:
                idf1s_detector = []
                for min_width, min_height in zip(min_widths, min_heights):
                    idf1s = week5.task1(
                        save_path=save_path,
                        distance_thresholds=distance_thresholds,
                        min_track_len=min_track_len,
                        min_width=min_width,
                        min_height=min_height,
                        sequence=sequence,
                        camera=camera,
                        detector=detector)

                    idf1s_detector.append(idf1s)

                all_idf1s.append(idf1s_detector)

            for idf1s in all_idf1s:
                plt.plot(min_widths, idf1s)
            plt.xticks([w for w in min_widths if w % 40 == 0])
            plt.xlabel('Minimum width (length = 0.8 * width)')
            plt.ylabel('IDF1')
            plt.legend(detectors, loc='best')
            if save_path:
                plt.savefig(os.path.join(save_path, 'task1_' + sequence + '_' + camera + '_' + 'min-width-length_vs_idf1.png'))
            plt.show()

        elif test_type == 'one_detector_all_cameras':

            ## No optical flow
            save_path = 'results/week5/task_1'
            distance_thresholds = [675]
            min_track_len = 5
            min_width = 60
            min_height = 48
            sequence = 'S03'
            cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
            detector = 'yolo3'

            all_idf1s = []
            for camera in cameras:
                idf1s = week5.task1(
                    save_path=save_path,
                    distance_thresholds=distance_thresholds,
                    min_track_len=min_track_len,
                    min_width=min_width,
                    min_height=min_height,
                    sequence=sequence,
                    camera=camera,
                    detector=detector)

                all_idf1s.append(idf1s[0])

            print(f'IDF1s for detector {detector} (mean IDF1 {np.mean( np.array(all_idf1s))}):')
            for idf1s, camera in zip(all_idf1s, cameras):
                print(f'\tcamera {camera}: {idf1s}')

            ## Optical flow
            # save_path = 'results/week5/task_1_optical_flow'
            # distance_thresholds = [675]
            # min_track_len = 5
            # min_width = 60
            # min_height = 48
            # sequence = 'S04'
            # cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
            # detector = 'yolo3'

            # all_idf1s = []
            # for camera in cameras:
            #     idf1s = week5.task1_optical_flow(
            #         save_path=save_path,
            #         distance_thresholds=distance_thresholds,
            #         min_track_len=min_track_len,
            #         min_width=min_width,
            #         min_height=min_height,
            #         sequence=sequence,
            #         camera=camera,
            #         detector=detector)

            #     all_idf1s.append(idf1s[0])

            # print(f'IDF1s for detector {detector} (mean IDF1 {np.mean( np.array(all_idf1s))}):')
            # for idf1s, camera in zip(all_idf1s, cameras):
            #     print(f'\tcamera {camera}: {idf1s}')

            ## Kalman filter
            # save_path = 'results/week5/task_1_kalman_filter'
            # distance_thresholds = [675]
            # min_track_len = 5
            # min_width = 60
            # min_height = 48
            # sequence = 'S04'
            # cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
            # detector = 'mask_rcnn'

            # all_idf1s = []
            # for camera in cameras:
            #     idf1s = week5.task1_kalman_filter(
            #         save_path=save_path,
            #         distance_thresholds=distance_thresholds,
            #         min_track_len=min_track_len,
            #         min_width=min_width,
            #         min_height=min_height,
            #         sequence=sequence,
            #         camera=camera,
            #         detector=detector)

            #     all_idf1s.append(idf1s[0])

            # print(f'IDF1s for detector {detector} (mean IDF1 {np.mean( np.array(all_idf1s))}):')
            # for idf1s, camera in zip(all_idf1s, cameras):
            #     print(f'\tcamera {camera}: {idf1s}')

        elif test_type == 'mean_idf1_across_cameras_sequence_03':

            save_path = 'results/week5/task_1'
            distance_thresholds = [400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800]
            min_track_len = 5
            min_width = 60
            min_height = 48
            sequence = 'S03'
            cameras = [os.path.split(p)[-1] for p in glob('data/AICity_data/train/' + sequence + '/c*')]
            detectors = ['mask_rcnn', 'ssd512', 'yolo3']

            all_idf1s = []
            for detector in detectors:
                idf1s_detector = []
                for camera in cameras:
                    idf1s = week5.task1(
                        save_path=save_path,
                        distance_thresholds=distance_thresholds,
                        min_track_len=min_track_len,
                        min_width=min_width,
                        min_height=min_height,
                        sequence=sequence,
                        camera=camera,
                        detector=detector)

                    idf1s_detector.append(idf1s)

                idf1s_detector = list( np.mean( np.array(idf1s_detector), axis=0 ) )

                all_idf1s.append(idf1s_detector)

            for idf1s, detector in zip(all_idf1s, detectors):
                plt.plot(distance_thresholds, idf1s)
            plt.xticks([d for d in distance_thresholds if d % 50 == 0])
            plt.xlabel('distance_thresholds')
            plt.ylabel('Mean IDF1 across cameras')
            plt.legend(detectors, loc='best')
            if save_path:
                plt.savefig(os.path.join(save_path, 'task1_mean-idf1-across-cameras-sequence-03.png'))
            plt.show()

    elif args.task == 2:
        week5.task2()
    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1, 2]")

else:
    raise ValueError(f"Bad input week {args.week}. Options are [1,2,3,4,5]")
