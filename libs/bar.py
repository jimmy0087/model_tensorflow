import time
import sys

def report_progress(progress, total, lbar_prefix = '', rbar_prefix='',dis_str=''):
    percent = round(progress / float(total) * 25)
    buf = "%s|%s|  %s%d/%d %s %s"%(lbar_prefix, ('>' * percent).ljust(25, '-'),
        rbar_prefix, progress, float(total), "%d%%"%(percent),dis_str)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()

def report_progress_done():
    sys.stdout.write('\n')

if __name__=='__main__':
    total = 100
    report_progress(0, total)
    for progress in range(1, 101):
        time.sleep(0.1)
        report_progress(progress, total)
    report_progress_done()