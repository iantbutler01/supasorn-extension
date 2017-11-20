import random
import glob
import os
import numpy as np
import struct
import bisect

def readSingleInt(path):
  with open(path) as f:
    return int(f.readline())

def readCVFloatMat(fl):
  with open(fl, 'rb') as f:
      t = struct.unpack('B', f.read(1))[0]
      if t != 5:
        return 0
      h = struct.unpack('i', f.read(4))[0]
      w = struct.unpack('i', f.read(4))[0]
      return np.reshape(np.array(struct.unpack('%df' % (h * w), f.read(4 * h * w)), float), (h, w))

def _str_to_bool(s):
  if s.lower() not in ['true', 'false']:
      raise ValueError('Need bool; got %r' % s)
  return s.lower() == 'true'

class DataLoader():

  def __init__(self, training_dir):
    self.training_dir = training_dir
    self.fps = 29

  def createInputFeature(self, audio, audiodiff, timestamps, startframe, nframe):
    startAudio = bisect.bisect_left(timestamps, (startframe - 1) // self.fps)
    endAudio = bisect.bisect_right(timestamps, (startframe + nframe - 2) // self.fps)
    audio_debug = (audio[startAudio:endAudio, :-1], audiodiff[startAudio:endAudio, :])
    print(audio_debug[0].shape, audio_debug[1].shape)
    inp = np.concatenate((audio[startAudio:endAudio, :-1], audiodiff[startAudio:endAudio, :]), axis=1)
    return startAudio, endAudio, inp 


  def load_preprocessed(self, inps, outps):
    newinps = {"training": [], "validation": []}
    newoutps = {"training": [], "validation": []}
    for key in newinps:
      for i in range(len(inps[key])):
        if len(inps[key][i]) - self.args.timedelay >= (self.args.seq_length+2):
          if self.args.timedelay > 0:
            newinps[key].append(inps[key][i][self.args.timedelay:])
            newoutps[key].append(outps[key][i][:-self.args.timedelay])
          else:
            newinps[key].append(inps[key][i])
            newoutps[key].append(outps[key][i])
    print("load preprocessed", len(newinps), len(newoutps))
    return newinps, newoutps

  def preprocess(self, save_dir):
    files = [x.split("\t")[0].strip() for x in open(self.training_dir + "processed_fps.txt", "r").readlines()]

    inps = {"training": [], "validation": []}
    outps = {"training": [], "validation": []}

    # validation = 0.2
    validation = 0
    for i in range(len(files)):
      tp = "training" if random.random() > validation else "validation"

      dnums = sorted([os.path.basename(x) for x in glob.glob(self.training_dir + files[i] + "}}*")])

      audio = np.load(self.training_dir + "/audio/normalized-cep13/" + files[i] + ".wav.npy") 
      audiodiff = audio[1:,:-1] - audio[:-1, :-1]

      print(files[i], audio.shape, tp)
      timestamps = audio[:, -1]

      for dnum in dnums:
        try:
            print(dnum)
            fids = readCVFloatMat(self.training_dir + dnum + "/frontalfidsCoeff_unrefined.bin")
            if not os.path.exists(self.training_dir + dnum + "/startframe.txt"):
              startframe = 1
            else:
              startframe = readSingleInt(self.training_dir + dnum + "/startframe.txt")
            nframe = readSingleInt(self.training_dir + dnum + "/nframe.txt")

            startAudio, endAudio, inp = self.createInputFeature(audio, audiodiff, timestamps, startframe, nframe)

            outp = np.zeros((endAudio - startAudio, fids.shape[1]), dtype=np.float32)
            leftmark = 0
            for aud in range(startAudio, endAudio):
              audiotime = audio[aud, -1]
              while audiotime >= (startframe - 1 + leftmark + 1) / self.fps:
                leftmark += 1
              t = (audiotime - (startframe - 1 + leftmark) / self.fps) * self.fps;
              outp[aud - startAudio, :] = fids[leftmark, :] * (1 - t) + fids[min(len(fids) - 1, leftmark + 1), :] * t;
                
            inps[tp].append(inp)
            outps[tp].append(outp)
        except Exception as e:
            print('Off by one error on last section, im not certain about.')

    return (inps, outps)

  def normalize(self, inps, outps):
    meani, stdi = normalizeData(inps["training"], "save/" + self.args.save_dir, "statinput", ["fea%02d" % x for x in range(inps["training"][0].shape[1])], normalize=self.args.normalizeinput)
    meano, stdo = normalizeData(outps["training"], "save/" + self.args.save_dir, "statoutput", ["fea%02d" % x for x in range(outps["training"][0].shape[1])], normalize=self.args.normalizeoutput)

    for i in range(len(inps["validation"])):
      inps["validation"][i] = (inps["validation"][i] - meani) / stdi;

    for i in range(len(outps["validation"])):
      outps["validation"][i] = (outps["validation"][i] - meano) / stdo;

    return meani, stdi, meano, stdo

  def loadData(self):
    if not os.path.exists("save/"):
      os.mkdir("save/")
    if not os.path.exists("save/" + self.args.save_dir):
      os.mkdir("save/" + self.args.save_dir)

    if len(self.args.usetrainingof):
      data_file = "data/training_" + self.args.usetrainingof + ".cpkl"
    else:
      data_file = "data/training_" + self.args.save_dir + ".cpkl"

    if not (os.path.exists(data_file)) or self.args.reprocess:
      print("creating training data cpkl file from raw source")
      inps, outps = self.preprocess(data_file)

      meani, stdi, meano, stdo = self.normalize(inps, outps)

      if not os.path.exists(os.path.dirname(data_file)):
        os.mkdir(os.path.dirname(data_file))
      f = open(data_file, "wb")
      cPickle.dump({"input": inps["training"], "inputmean": meani, "inputstd": stdi, "output": outps["training"], "outputmean":meano, "outputstd": stdo, "vinput": inps["validation"], "voutput": outps["validation"]}, f, protocol=2) 
      f.close() 


    f = open(data_file,"rb")
    data = cPickle.load(f)
    inps = {"training": data["input"], "validation": data["vinput"]} 
    outps = {"training": data["output"], "validation": data["voutput"]} 
    f.close()

    self.dimin = inps["training"][0].shape[1]
    self.dimout = outps["training"][0].shape[1]

    self.inps, self.outps = self.load_preprocessed(inps, outps)


  def sample(self, sess, args, data, pt):
    if self.audioinput:
      self.sample_audioinput(sess, args, data, pt)
    else:
      self.sample_videoinput(sess, args, data, pt)

  def sample_audioinput(self, sess, args, data, pt):
    meani, stdi, meano, stdo = data["inputmean"], data["inputstd"], data["outputmean"], data["outputstd"]
    audio = np.load(self.training_dir + "/audio/normalized-cep13/" + self.args.input2 + ".wav.npy") 

    audiodiff = audio[1:,:-1] - audio[:-1, :-1]
    timestamps = audio[:, -1]

    times = audio[:, -1]
    inp = np.concatenate((audio[:-1, :-1], audiodiff[:, :]), axis=1)

    state = []
    for c, m in self.initial_state: # initial_state: ((c1, m1), (c2, m2))
      state.append((c.eval(), m.eval()))

    if not os.path.exists("results/"):
      os.mkdir("results/")

    f = open("results/" + self.args.input2 + "_" + args.save_dir + ".txt", "w")
    print("output to results/" + self.args.input2 + "_" + args.save_dir + ".txt")
    f.write("%d %d\n" % (len(inp), self.dimout + 1))
    fetches = []
    fetches.append(self.output)
    for c, m in self.final_state: # final_state: ((c1, m1), (c2, m2))
      fetches.append(c)
      fetches.append(m)

    feed_dict = {}
    for i in range(len(inp)):
      for j, (c, m) in enumerate(self.initial_state):
        feed_dict[c], feed_dict[m] = state[j]

      input = (inp[i] - meani) / stdi
      feed_dict[self.input_data] = [[input]]
      res = sess.run(fetches, feed_dict)
      output = res[0] * stdo + meano

      if i >= args.timedelay:
        shifttime = times[i - args.timedelay]
      else:
        shifttime = times[0]
      f.write(("%f " % shifttime) + " ".join(["%f" % x for x in output[0]]) + "\n")

      state_flat = res[1:]
      state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)] 
    f.close()
