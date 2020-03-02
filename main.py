import argparse
import os

from week1 import task1, task2, task3_4

parser = argparse.ArgumentParser(description='M6 - Video Analysis: Video Surveillance for Road Traffic Monitoring')

parser.add_argument('-w', '--week', type=int, help='week to execute. Options are [1]')
parser.add_argument('-t', '--task', type=int, help='task to execute. Options are [1,2,3,4]')
args = parser.parse_args()

path_plots = 'results/'
if not os.path.exists(path_plots):
    os.makedirs(path_plots)

if args.week == 1:
    if args.task == 1:
        task1(path_plots)
    elif args.task == 2:
        task2(path_plots)
    elif args.task == 3 or args.task == 4:
        task3_4(path_plots)
    else:
        raise ValueError(f"Bad input task {args.task}. Options are [1,2,3,4]")
else:
    raise ValueError(f"Bad input weelk {args.week}. Options are [1]")