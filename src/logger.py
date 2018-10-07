import logging
import os
import datetime
import sys
import types

class RolloverHandler(logging.FileHandler):
    def __init__(self, logfile, *args, **kargs):
        if os.path.exists(logfile):
            dirname = os.path.dirname(os.path.abspath(logfile))
            basename = os.path.basename(logfile)
            bare_name, ext = os.path.splitext(basename)
            timestamp = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
            new_logfile = os.path.sep.join([dirname, "%s_%s%s" % (bare_name, timestamp, ext)])
            os.rename(logfile, new_logfile)
        logging.FileHandler.__init__(self, logfile, *args, **kargs)

def init_logger(logfile, loglevel=logging.DEBUG):
    def traceback(self, fault):
        msg = "Error %s:%s. Traceback -" % (str(fault.__class__), str(fault))
        import traceback as tr
        msg += "".join(tr.format_exception(*sys.exc_info()))
        self.error("%s", msg)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(funcName)s:%(lineno)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RolloverHandler(logfile, mode = "w")
    fh.setFormatter(formatter)
    fh.setLevel(loglevel)

    log = logging.getLogger()
    setattr(log, "traceback", types.MethodType(traceback, log))
    log.addHandler(fh)
    log.setLevel(loglevel)

    return log
