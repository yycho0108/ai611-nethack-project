# AI611 Nethack Project

Project Repository for [Nethack Challenge](https://nethackchallenge.com/) .

AI611: Deep Reinforcement Learning, KAIST GSAI 2022SP

## How to train the agent

First, run the docker container:

```bash
$ ./docker/run.sh
```

Then, install additional dependencies
within the docker container:
```bash
$ python3 -m pip install stable_baselines tensorboard
```
(Note that this step is needed, since we do not build a custom docker image,
but use the image provided by `fairnle/challenge:dev`.)

Afterwards, you can freely train our agent:

```bash
$ python3 train_pop_art_agent.py
```

## How to record the video

After training the agent, you can record the video as follows:

```bash
$ python3 record.video.py ${CKPT_FILE} # e.g. '/tmp/nethack/run-022/log/nh-pa-last.pt'
$ python3 -m nle.scripts.ttyplay /tmp/record/nle.PPPPP.I.ttyrec.bz2
```

Where `PPPPP` is the process ID and `I` is the environment index.
In general, you'd want to set `PPPPP` to the latest one, and `I = 0`.

# Collaborators

* Yoonyoung(Jamie) Cho[Github](https://github.com/yycho0108)[Website](https://yycho0108.github.io)
* Minchan Kim[Github](https://github.com/JoHeeSang)
* Heesang Jo[Github](https://github.com/Minchan-Kim)
