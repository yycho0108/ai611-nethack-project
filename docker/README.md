# Docker

Docker configuration for Nethack project.

## Running the image

Running the below command pulls the docker image
from [Docker Hub](https://hub.docker.com/) locally,
and runs the docker container with reasonable defaults
(gpu/network/privileged) with our git repository
mounted in `${HOME}`.

```bash
$ ./run.sh
```

To test if it works, inside the docker container:

```bash
$ python3 -m nle.scripts.play --mode random
```

You should see an output like the following:

```bash
Available actions: (<MiscAction.MORE: 13>, <CompassDirection.N: 107>, <CompassDirection.E: 108>, <CompassDirection.S: 106>, <CompassDirection.W: 104>, <CompassDirection.NE: 117>, <CompassDirection.SE: 110>, <CompassDirection.SW: 98>, <CompassDirection.NW: 121>, <CompassDirectionLonger.N: 75>, <CompassDirectionLonger.E: 76>, <CompassDirectionLonger.S: 74>, <CompassDirectionLonger.W: 72>, <CompassDirectionLonger.NE: 85>, <CompassDirectionLonger.SE: 78>, <CompassDirectionLonger.SW: 66>, <CompassDirectionLonger.NW: 89>, <MiscDirection.UP: 60>, <MiscDirection.DOWN: 62>, <MiscDirection.WAIT: 46>, <Command.KICK: 4>, <Command.EAT: 101>, <Command.SEARCH: 115>)
--------
Previous reward: -0.01
Previous action: <MiscAction.MORE: 13>
--------




                        ----------
                       #.........-######
                       #|........|     #
                        |.......f|
                        |...@....|
                        |.......)|
                        -----|----
                             ###
                               #
                               ##
                           -----.--
                           |......|
                           |..[...-#
                           |.....<|
                           --------




Agent the Candidate            St:17 Dx:16 Co:10 In:11 Wi:13 Ch:8 Neutral S:0
Dlvl:1 $:0 HP:14(14) Pw:4(4) AC:4 Xp:1/0
--------

```

Beware that running the example may create `nle_data/` directory in the `${PWD}`.
