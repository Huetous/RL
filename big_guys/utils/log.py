# import os.path as osp
# import os, time, json, atexit, csv
# from import convert_json
# import joblib
# import numpy as np
#
# color2num = dict(
#     gray=30,
#     red=31,
#     green=32,
#     yellow=33,
#     blue=34,
#     magenta=35,
#     cyan=36,
#     white=37,
#     crimson=38
# )
#
#
# def colorize(string, color, bold=False, highlight=False):
#     attr = []
#     num = color2num[color]
#     if highlight:
#         num += 10
#     attr.append(str(num))
#     if bold:
#         attr.append('1')
#     return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
#
#
# class Logger:
#     def __init__(self, output_dir=None, output_fname="progress.txt", exp_name=None):
#         self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
#         if osp.exists(self.output_dir):
#             print("Warning: Log dir %s already exists! Storing info there anyway." % self.output_dir)
#         else:
#             os.makedirs(self.output_dir)
#         self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
#         atexit.register(self.output_file.close)
#         print(colorize("Logging data to %s" % self.output_file.name, "green", bold=True))
#
#         self.first_row = True
#         self.log_headers = []
#         self.log_current_row = {}
#         self.exp_name = exp_name
#
#         self.data = []
#
#     def log(self, msg, color="green"):
#         print(colorize(msg, color, bold=True))
#
#     def log_tabular(self, key, val):
#         if self.first_row:
#             self.log_headers.append(key)
#         else:
#             assert key in self.log_headers, "Unexpected new key %s." % key
#         assert key not in self.log_current_row, "You already set %s this iteration." % key
#         self.log_current_row[key] = val
#
#     def save_config(self, config):
#         config_json = convert_json(config)
#         if self.exp_name is not None:
#             config_json["exp_name"] = self.exp_name
#         output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
#         print(colorize("Saving config:\n", color="cyan", bold=True))
#         print(output)
#         with open(osp.join(self.output_dir, "config.json"), 'w') as out:
#             out.write(output)
#
#     def save_state(self, state_dict, itr=None):
#         fname = "vars.pkl" if itr is None else "vars%d.pkl" % itr
#         try:
#             joblib.dump(state_dict, osp.join(self.output_dir, fname))
#         except:
#             self.log("Warning: could not pickle state_dict.", color="red")
#
#     def dump_tabular(self):
#         vals = []
#         key_lens = [len(key) for key in self.log_headers]
#         max_key_len = max(15, max(key_lens))
#         keystr = '%' + '%d' % max_key_len
#         fmt = "| " + keystr + "s | %15s |"
#         n_slashes = 22 + max_key_len
#         print("-" * n_slashes)
#         for key in self.log_headers:
#             val = self.log_current_row.get(key, "")
#             valstr = "%8.3g" % val if hasattr(val, "__float__") else val
#             print(fmt % (key, valstr))
#             vals.append(val)
#         self.data.append(vals)
#         print("-" * n_slashes, flush=True)
#         if self.output_file is not None:
#             if self.first_row:
#                 self.output_file.write("\t".join(self.log_headers) + "\n")
#             self.output_file.write("\t".join(map(str, vals)) + "\n")
#             self.output_file.flush()
#
#         self.log_current_row.clear()
#         self.first_row = False
#
#     def save_to_csv(self):
#         with open(self.output_dir + "/progress.csv", "wt") as fp:
#             writer = csv.writer(fp, delimiter=",")
#             writer.writerow(self.log_headers)
#             writer.writerows(self.data)
#
#
# class EpochLogger(Logger):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.epoch_dict = dict()
#
#     def store(self, **kwargs):
#         for k, v in kwargs.items():
#             if not (k in self.epoch_dict.keys()):
#                 self.epoch_dict[k] = []
#             self.epoch_dict[k].append(v)
#
#
#     def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
#         if val is not None:
#             super().log_tabular(key, val)
#         else:
#             v = self.epoch_dict[key]
#             super().log_tabular(key if average_only else 'Average' + key, np.mean(v))
#             if not average_only:
#                 super().log_tabular('Std' + key, np.std(v))
#             if with_min_and_max:
#                 super().log_tabular('Max' + key, np.max(v))
#                 super().log_tabular('Min' + key, np.min(v))
#         self.epoch_dict[key] = []
#
#     def get_stats(self, key):
#         v = self.epoch_dict[key]
#         vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
#         return vals
