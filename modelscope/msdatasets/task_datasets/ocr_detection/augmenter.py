# ------------------------------------------------------------------------------
# The implementation is adopted from DBNet,
# made publicly available under the Apache License 2.0 at https://github.com/MhLiao/DB.
# ------------------------------------------------------------------------------
import imgaug
import imgaug.augmenters as iaa


class AugmenterBuilder(object):

    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None:
            return None
        elif isinstance(args, (int, float, str)):
            return args
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(
                    iaa,
                    args[0])(*[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(
                    **{
                        k: self.to_tuple_if_list(v)
                        for k, v in args.items() if not k == 'cls'
                    })
            else:
                return {
                    key: self.build(value, root=False)
                    for key, value in args.items()
                }
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj
