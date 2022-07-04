import cProfile
import inspect
import traceback

def pyca_profile(func):
    # example, run in D:\Dropbox\Python\Pyca\profiler:
    #   snakeviz run_analysis.prof
    def wrapper_profile(self, *args, **kwargs):
        fn = 'profiler/'+func.__name__+'.prof'
        # pr = cProfile.Profile()
        # pr.enable()
        # try:
        #     val = func(self, *args, **kwargs)
        # except TypeError:
        #     try:
        #         val = func(self, *args)
        #     except TypeError:
        #         try:
        #             val = func(self, **kwargs)
        #         except TypeError:
        #             try:
        #                 val = func(self)
        #             except:
        #                 print(func.__name__ + " couldn't be profiled.")

        # pr.disable()
        # open(fn, 'a').close()
        # pr.dump_stats(fn)
        args = [self] + list(args)
        prof = cProfile.Profile()
        val = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(fn)
        return val
    return wrapper_profile


class handle_error(object):
    def __init__(self, callback_attributes=[], msg_prefix="", msg_suffix="", msg=""):
        self.callback_attributes = callback_attributes
        self.msg_prefix = msg_prefix
        self.msg_suffix = msg_suffix
        self.msg        = msg

    def __call__(self, f):
        def wrapper(*args, **kwargs):
            try:
                f(*args, **kwargs)
            except Exception as e:
                for callback_attribute in self.callback_attributes:
                    try:
                        # Get attribute (method) from the class whose method
                        # was decorated with this decorator
                        callback = getattr(args[0], callback_attribute)
                    except AttributeError as ae:
                        print(str(callback_attribute) + ' not found on ' + str(args[0]))
                        print(ae)

                    if callable(callback):
                        try:
                            if self.msg == "":
                                callback(self.msg_prefix + str(e) + self.msg_suffix)
                            else:
                                callback(self.msg)
                        except TypeError as te:
                            print('callback method must accept 1 argument, a string.')
                            print(te)
                traceback.print_exc()
                print(str(e) + " -- occurred on " + f.__name__)
        return wrapper

