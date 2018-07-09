from __future__ import print_function

import sys
import time
#try:
#    from IPython.core.display import HTML, Javascript, display
#except ImportError:
#    pass

__all__ = ['progress_bar']


class ProgressBar(object):

    def __init__(self, iterations, animation_interval=.5):
        self.iterations = iterations
        self.start = time.time()
        self.last = 0
        self.animation_interval = animation_interval

    def percentage(self, i):
        return 100 * i / float(self.iterations)

    def update(self, i):
        elapsed = time.time() - self.start
        i = i + 1

        if elapsed - self.last > self.animation_interval:
            self.animate(i + 1, elapsed)
            self.last = elapsed
        elif i == self.iterations:
            self.animate(i, elapsed)

class TextProgressBar(ProgressBar):

    def __init__(self, iterations, printer):
        self.fill_char = '-'
        self.width = 40
        self.printer = printer

        ProgressBar.__init__(self, iterations)
        self.update(0)

    def animate(self, i, elapsed):
        self.printer(self.progbar(i, elapsed))

    def progbar(self, i, elapsed):
        bar = self.bar(self.percentage(i))
        return "[%s] %i of %i complete in %.1f sec" % (
            bar, i, self.iterations, round(elapsed, 1))

    def bar(self, percent):
        all_full = self.width - 2
        num_hashes = int(percent / 100 * all_full)

        bar = self.fill_char * num_hashes + ' ' * (all_full - num_hashes)

        info = '%d%%' % percent
        loc = (len(bar) - len(info)) // 2
        return replace_at(bar, info, loc, loc + len(info))


def replace_at(dstr, new, start, stop):
    return dstr[:start] + new + dstr[stop:]


def consoleprint(s):
    if check_windows_platform():
        print(s, '\r', end='')
    else:
        print(s)

def check_windows_platform():
    return sys.platform.lower().startswith('win')

def ipythonprint(s, flush = True):
    print('\r', s, end='')
    if flush is True:
        flush_print()

def flush_print():
    sys.stdout.flush()

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def progress_bar(iters):
    '''
    Simulation progress bar.

    A simple progress bar to monitor MCMC sampling progress.
    Modified from original code by Corey Goldberg (2010).

    Args:
        * **iters** (:py:class:`int`): Number of iterations in simulation.

    Example display:

    ::

        [--------         21%                  ] 2109 of 10000 complete in 0.5 sec

    .. note::

        Will display a progress bar as simulation runs, providing
        feedback as to the status of the simulation.  Depending on the available
        resources, the appearance of the progress bar may differ.
    '''
    if run_from_ipython():
        return TextProgressBar(iters, ipythonprint)
    else:
        return TextProgressBar(iters, consoleprint)