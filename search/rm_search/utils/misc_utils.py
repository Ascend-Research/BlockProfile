import collections


class UniqueDict(dict):
    def __init__(self, inp=None):
        self._no_dups = True
        if isinstance(inp, dict):
            super(UniqueDict,self).__init__(inp)
        else:
            super(UniqueDict,self).__init__()
            if isinstance(inp, (collections.Mapping, collections.Iterable)):
                si = self.__setitem__
                for k,v in inp:
                    si(k,v)
        self._no_dups = False

    def __setitem__(self, k, v):
        try:
            self.__getitem__(k)
            if self._no_dups:
                raise ValueError("duplicate key '{0}' found".format(k))
            else:
                super(UniqueDict, self).__setitem__(k, v)
        except KeyError:
            super(UniqueDict,self).__setitem__(k,v)


class RunningStatMeter(object):

    def __init__(self):
        self.avg = 0.
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0.
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        return self

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.max = max(self.max, val)
        self.min = min(self.min, val)


def load_lat_data_from_csv(fp, lat_key="overall npu lat"):
    # net_id,net_str,device,resolution,lat unit,end-to-end lat
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = lines[1:]
    data = []
    for line in lines:
        line = line.strip()
        if len(line) == 0: continue
        net_id, net_str, device, resolution, \
            lat_unit, overall_lat = line.split(",")
        net_config = []
        stages = net_str.split("|")
        for stage in stages:
            blocks = stage.split("->")
            net_config.append(blocks)
        data.append({
            "net_id": int(net_id),
            "net": net_config,
            "device": device,
            "lat_unit": lat_unit,
            "resolution": int(resolution),
            lat_key: float(overall_lat),
        })
    return data
