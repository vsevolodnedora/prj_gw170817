#
#



#resolutions = {"HR":123., "SR":185.,"LR":246.}
resolutions = {"HR":0.083333, "SR":0.125,"LR":0.166666}

simulations = {
    "BLh":
        {    # q=1.8                 # 20 ms | BH | 3D | PC     # 25 ms | BH | 3D(4) | PC  # 27 ms | BH | 3D(3) | PC
            "BLh_M10201856_M0_LK": ["BLh_M10201856_M0_LK_HR", "BLh_M10201856_M0_LK_LR", "BLh_M10201856_M0_LK_SR"],
             # q=1.8            # 57 ms | BH | no3D    # 65 ms | BH | 3D(7)   # 37 ms | BH | 3D (5) | PC
            "BLh_M10201856_M0":["BLh_M10201856_M0_HR", "BLh_M10201856_M0_LR", "BLh_M10201856_M0_SR"],
            # q = 1.7                # 30 ms | stable | 3D      # 74 ms | stable | 3D |
            "BLh_M10651772_M0_LK":["BLh_M10651772_M0_LK_SR", "BLh_M10651772_M0_LK_LR"],
            # q = 1.5               # 56ms | stable | 3D
            "BLh_M11041699_M0_LK": ["BLh_M11041699_M0_LK_LR"],
            # q = 1.5               # 27/40 ms | stable | 3D |missing
            "BLh_M11041699_M0": ["BLh_M11041699_M0_LR"],
            # q = 1.4               # 71ms | stable | 3D       # 49 | stable (wrong merger time) |
            "BLh_M11461635_M0_LK": ["BLh_M11461635_M0_LK_SR", "BLh_M16351146_M0_LK_LR"],
            # q = 1.3               # 75ms | stable | no3D     # 21ms | stable | 3D(5)
            "BLh_M11841581_M0_LK": ["BLh_M11841581_M0_LK_LR", "BLh_M11841581_M0_LK_SR"],
            # q=1.3             # 28 ms | stable | 3D
            "BLh_M11841581_M0": ["BLh_M11841581_M0_LR"],
            # q = 1.2           # 27ms | stable | 3D
            "BLh_M12591482_M0": ["BLh_M12591482_M0_LR"],
            # q = 1.2              # 81ms | stable | no3D
            "BLh_M12591482_M0_LK":["BLh_M12591482_M0_LK_LR"],
            # q = 1.                # 102 ms | stable | 3D     # 54 ms | stable | 3D(7)
            "BLh_M13641364_M0_LK": ["BLh_M13641364_M0_LK_SR", "BLh_M13641364_M0_LK_LR"],
            # q = 1.            # 47 ms | stable | 3D
            "BLh_M13641364_M0": ["BLh_M13641364_M0_LR"]
        },
    "DD2":
        {
            # q = 1.            # 50ms | stable | 3D    # 110ms | stable | 3D   # 19 ms | stable | no3D
            "DD2_M13641364_M0": ["DD2_M13641364_M0_LR", "DD2_M13641364_M0_SR", "DD2_M13641364_M0_HR"],
            # q = 1.                 # 101ms | stable | 3D(3)    # 120ms | stable | 3D(5)   # 17ms | stable | no3D
            "DD2_M13641364_M0_R04": ["DD2_M13641364_M0_LR_R04", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_HR_R04"],
            # q = 1.                    # 130ms | stable | 3D(3)       # 120ms | stable | 3D(51ms+) # 82ms | stable | 3D
            "DD2_M13641364_M0_LK_R04": ["DD2_M13641364_M0_LK_LR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_HR_R04"],
            # q = 1.1           # 51ms | stable | 3D
            "DD2_M14321300_M0": ["DD2_M14321300_M0_LR"],
            # q = 1.1           # 38ms | stable | 3D
            "DD2_M14351298_M0": ["DD2_M14351298_M0_LR"],
            # q = 1.2           # 40ms | stable | 3D    # 70ms | stable | 3D
            "DD2_M14861254_M0": ["DD2_M14861254_M0_LR", "DD2_M14861254_M0_HR"],
            # q = 1.2            # 47ms | stable | 3D   # 99ms | stable | 3D   # 63ms | stable | 3D
            "DD2_M14971246_M0": ["DD2_M14971246_M0_LR", "DD2_M14971245_M0_SR", "DD2_M14971245_M0_HR"],
            # q = 1.2               # 113ms | stable | 3D(60ms+) # 28ms | stable | no3D
            "DD2_M15091235_M0_LK": ["DD2_M15091235_M0_LK_SR", "DD2_M15091235_M0_LK_HR"],
            # q = 1.4               # # 47ms | stable | 3D     # 65ms | stable | 3D (wrong init.data)
            "DD2_M11461635_M0_LK":["DD2_M16351146_M0_LK_LR"]    #, "DD2_M11461635_M0_LK_SR"]
        },
    "LS220":
        {
            # q = 1.                # 50ms | BH | 3D          # 41ms | BH | no3D      # 49ms | BH | 3D | wrong BH time
            "LS220_M13641364_M0": ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_HR", "LS220_M13641364_M0_LR"],
            # q = 1.                 # 38ms | BH | 3D
            "LS220_M13641364_M0_LK": ["LS220_M13641364_M0_LK_SR_restart"],
            # q = 1.1               # 38ms | BH | no3D        # 37ms | BH | no3D
            "LS220_M14001330_M0": ["LS220_M14001330_M0_HR", "LS220_M14001330_M0_SR"],
            # q = 1.1              # 38ms | stable | no 3D    # 39ms | BH | no3D
            "LS220_M14351298_M0":["LS220_M14351298_M0_HR", "LS220_M14351298_M0_SR"],
            # q = 1.2             # 49ms | stable | no3D   # 43ms | BH | no3D       # 43ms | stable | np3D
            "LS220_M14691268_M0":["LS220_M14691268_M0_SR", "LS220_M14691268_M0_HR", "LS220_M14691268_M0_LR"],
            # q = 1.2                # 24ms | stable | no3D       # 107ms | long-lived BH | 3D(60ms+)
            "LS220_M14691268_M0_LK":["LS220_M14691268_M0_LK_HR", "LS220_M14691268_M0_LK_SR"],
            # q = 1.4               # 30ms | BH | 3D            # missing 38ms | BH | 3D
            "LS220_M16351146_M0_LK":["LS220_M16351146_M0_LK_LR", "LS220_M11461635_M0_LK_SR"],
            # q = 1.7               # missing 23 ms | BH | PC    # 16ms | BH | 3D | PC
            "LS220_M10651772_M0_LK": ["LS220_M10651772_M0_LK_SR", "LS220_M10651772_M0_LK_LR"]
        },
    "SFHo":
        {
            # q = 1.              # 22ms | BH | 3D        # 23ms | BH | 3D
            "SFHo_M13641364_M0": ["SFHo_M13641364_M0_SR", "SFHo_M13641364_M0_HR"],
            # q = 1.                # 29ms | BH | no3D
            "SFHo_M13641364_M0_LK": ["SFHo_M13641364_M0_LK_SR"],
            # q = 1.                        # 24ms | BH | no3D          # 37ms | BH | no3D
            "SFHo_M13641364_M0_LK_p2019":["SFHo_M13641364_M0_LK_HR", "SFHo_M13641364_M0_LK_SR_2019pizza"],
            # q = 1.1              # 29ms | BH | 3D        # 32ms | BH | 3D
            "SFHo_M14521283_M0":["SFHo_M14521283_M0_HR", "SFHo_M14521283_M0_SR"],
            # q = 1.1                       # 26ms | BH | no3D         # 26ms | BH | no3D
            "SFHo_M14521283_M0_LK_p2019":["SFHo_M14521283_M0_LK_HR", "SFHo_M14521283_M0_LK_SR_2019pizza"],
            # q = 1.1                   # 24ms | BH | no3D
            "SFHo_M14521283_M0_LK_SR":["SFHo_M14521283_M0_LK_SR"],
            # q = 1.4                   # 31ms | BH | 3D
            "SFHo_M16351146_M0_LK_p2019":["SFHo_M16351146_M0_LK_LR"]
            # q = 1.4                  # 65(missing5)ms | stable | 3D [WRONG INIT. DATA]
            # "SFHo_M11461635_M0_LK": ["SFHo_M11461635_M0_LK_SR"], # [WRONG INIT DATA]
            # q = 1.7                   # 21ms | BH | 3D | PC      # 26ms | stable | 3D [might be wrong]
            # "SFHo_M10651772_M0_LK": ["SFHo_M10651772_M0_LK_SR","SFHo_M10651772_M0_LK_LR"]

        },
    "SLy4":
        {
            # q = 1.                # 21ms | BH | no3D         # 24ms | BH | no3D
            "SLy4_M13641364_M0_LK":["SLy4_M13641364_M0_LK_LR", "SLy4_M13641364_M0_LK_SR"],
            # q = 1.            # 36ms | BH | 3D
            "SLy4_M13641364_M0":["SLy4_M13641364_M0_SR"],
            # q = 1.1             # 28ms | BH | extracting 3D # 34ms | BH | 3D
            "SLy4_M14521283_M0": ["SLy4_M14521283_M0_LR", "SLy4_M14521283_M0_SR"],
            # q = 1.4               # 67ms | stable | 3D [ WRONG INIT. DATA ]
            # "SLy4_M11461635_M0_LK": ["SLy4_M11461635_M0_LK_SR"],
            # q = 1.8               # 17ms | BH | 3D | PC
            "SLy4_M10201856_M0_LK": ["SLy4_M10201856_M0_LK_SR"]
        }
}

old_simulations = {
    "BHBlp":
        {
            "BHBlp_M125125_LK": ["BHBlp_M125125_LK"],
            "BHBlp_M1251365_LK": ["BHBlp_M1251365_LK"],
            "BHBlp_M130130_LK":["BHBlp_M130130_LK"],
            "BHBlp_M135135_LK":["BHBlp_M135135_LK", "BHBlp_M135135_LK_HR"],
            "BHBlp_M135135_M0":["BHBlp_M135135_M0"],
            "BHBlp_M140120_LK":["BHBlp_M140120_LK"],
            "BHBlp_M140120_M0":["BHBlp_M140120_M0"],
            "BHBlp_M140140_LK":["BHBlp_M140140_LK","BHBlp_M140140_LK_HR"],
            "BHBlp_M144139_LK":["BHBlp_M144139_LK"],
            "BHBlp_M150150_LK":["BHBlp_M150150_LK", "BHBlp_M150150_LK_HR"],
            "BHBlp_M160160_LK":["BHBlp_M160160_LK"]
        },
    "DD2":
        {
            "DD2_M120120_LK":["DD2_M120120_LK"],
            "DD2_M125125_LK":["DD2_M125125_LK"],
            "DD2_M1251365_LK":["DD2_M1251365_LK"],
            "DD2_M130130_LK":["DD2_M130130_LK"],
            "DD2_M135135_LK":["DD2_M135135_LK","DD2_M135135_LK_HR"],
            "DD2_M135135_M0":["DD2_M135135_M0"],
            "DD2_M140120_LK":["DD2_M140120_LK"],
            "DD2_M140120_M0":["DD2_M140120_M0"],
            "DD2_M140140_LK":["DD2_M140140_LK","DD2_M140140_LK_HR"],
            "DD2_M144139_LK":["DD2_M144139_LK"],
            "DD2_M150150_LK":["DD2_M150150_LK","DD2_M150150_LK_HR"],
            "DD2_M160160_LK":["DD2_M160160_LK"]
        },
    "LS220":
        {
            "LS220_M120120_LK":["LS220_M120120_LK"],
            "LS220_M1251365_LK":["LS220_M1251365_LK"],
            "LS220_M135135_LK":["LS220_M135135_LK_LR", "LS220_M135135_LK", "LS220_M135135_LK_HR"],
            "LS220_M135135_M0":["LS220_M135135_M0"],
            "LS220_M140120_LK":["LS220_M140120_LK"],
            "LS220_M140120_M0":["LS220_M140120_M0"],
            "LS220_M140140_LK":["LS220_M140140_LK_LR","LS220_M140140_LK","LS220_M140140_LK_HR"],
            "LS220_M144139_LK":["LS220_M144139_LK"],
            "LS220_M145145_LK":["LS220_M145145_LK"],
            "LS220_M150150_LK":["LS220_M150150_LK"],
            "LS220_M160160_LK":["LS220_M160160_LK"],
            "LS220_M171171_LK":["LS220_M171171_LK"]
        },
    "SFHo":
        {
            "SFHo_M1251365_LK":["SFHo_M1251365_LK"],
            "SFHo_M135135_LK":["SFHo_M135135_LK_LR", "SFHo_M135135_LK", "SFHo_M135135_LK_HR"],
            "SFHo_M135135_M0":["SFHo_M135135_M0"],
            "SFHo_M140120_LK":["SFHo_M140120_LK"],
            "SFHo_M140120_M0":["SFHo_M140120_M0"],
            "SFHo_M140140_LK":["SFHo_M140140_LK"],
            "SFHo_M144139_LK":["SFHo_M144139_LK"],
            "SFHo_M146146_LK":["SFHo_M146146_LK"]
        }
}


# simulations = {
#     "BLh":
#         {
#            "q=1.8":{
#                "BLh_M10201856_M0_LK":
#                    # 20 ms | BH | 3D | PC     # 25 ms | BH | 3D(4) | PC  # 27 ms | BH | 3D(3) | PC
#                    ["BLh_M10201856_M0_LK_HR", "BLh_M10201856_M0_LK_LR", "BLh_M10201856_M0_LK_SR"],
#                "BLh_M10201856_M0":
#                    # 57 ms | BH | no3D       # 65 ms | BH | 3D(7)      # 37 ms | BH | 3D (5) | PC
#                    ["BLh_M10201856_M0_HR", "BLh_M10201856_M0_LR", "BLh_M10201856_M0_SR"]
#            },
#            "q=1.7":{
#                "BLh_M10651772_M0_LK":
#                    # 30 ms | stable | 3D      # 74 ms | stable | 3D |
#                    ["BLh_M10651772_M0_LK_SR", "BLh_M10651772_M0_LK_LR"]
#            },
#            "q=1.5":{
#                "BLh_M11041699_M0_LK":
#                    # 56ms | stable | 3D
#                    ["BLh_M11041699_M0_LK_LR"],
#                "BLh_M11041699_M0":
#                    # 27/40 ms | stable | 3D |missing
#                    ["BLh_M11041699_M0_LR"]
#            },
#            "q=1.4":{
#                "BLh_M11461635_M0_LK":
#                    # 71ms | stable | 3D       # 49 | stable (wrong merger time) |
#                    ["BLh_M11461635_M0_LK_SR", "BLh_M16351146_M0_LK_LR"]
#            },
#            "q=1.3":{
#                "BLh_M11841581_M0_LK":
#                    # 75ms | stable | no3D     # 21ms | stable | 3D(5)
#                    ["BLh_M11841581_M0_LK_LR", "BLh_M11841581_M0_LK_SR"],
#                "BLh_M11841581_M0":
#                     # 28 ms | stable | 3D
#                    ["BLh_M11841581_M0_LR"]
#            },
#            "q=1.2":{
#                "BLh_M12591482_M0":
#                    # 27ms | stable | 3D
#                    ["BLh_M12591482_M0_LR"],
#                "BLh_M12591482_M0_LK":
#                    # 81ms | stable | no3D
#                    ["BLh_M12591482_M0_LK_LR"]
#            },
#            "q=1":{
#                "BLh_M13641364_M0_LK":
#                    # 102 ms | stable | 3D     # 54 ms | stable | 3D(7)
#                    ["BLh_M13641364_M0_LK_SR", "BLh_M13641364_M0_LK_LR"],
#                "BLh_M13641364_M0":
#                    # 47 ms | stable | 3D
#                    ["BLh_M13641364_M0_LR"]
#            }
#         },
#     "DD2":
#         {
#            "q=1":{
#                "DD2_M13641364_M0":
#                    # 50ms | stable | 3D    # 110ms | stable | 3D   # 19 ms | stable | no3D
#                    ["DD2_M13641364_M0_LR", "DD2_M13641364_M0_SR", "DD2_M13641364_M0_HR"],
#                "DD2_M13641364_M0_R04":
#                    # 101ms | stable | 3D(3)    # 120ms | stable | 3D(5)   # 17ms | stable | no3D
#                    ["DD2_M13641364_M0_LR_R04", "DD2_M13641364_M0_SR_R04", "DD2_M13641364_M0_HR_R04"],
#                "DD2_M13641364_M0_LK_R04":
#                     # 130ms | stable | 3D(3)       # 120ms | stable | 3D(51ms+) # 82ms | stable | 3D
#                    ["DD2_M13641364_M0_LK_LR_R04", "DD2_M13641364_M0_LK_SR_R04", "DD2_M13641364_M0_LK_HR_R04"]
#            },
#            "q=1.1":{
#                "DD2_M14321300_M0":
#                    # 51ms | stable | 3D
#                    ["DD2_M14321300_M0_LR"],
#                "DD2_M14351298_M0":
#                          # 38ms | stable | 3D
#                     ["DD2_M14351298_M0_LR"]
#            },
#            "q=1.2":{
#                "DD2_M14861254_M0":
#                     # 40ms | stable | 3D    # 70ms | stable | 3D
#                    ["DD2_M14861254_M0_LR", "DD2_M14861254_M0_HR"],
#                "DD2_M14971246_M0":
#                     # 47ms | stable | 3D   # 99ms | stable | 3D   # 63ms | stable | 3D
#                    ["DD2_M14971246_M0_LR", "DD2_M14971245_M0_SR", "DD2_M14971245_M0_HR"],
#                "DD2_M15091235_M0_LK":
#                    # 113ms | stable | 3D(60ms+) # 28ms | stable | no3D
#                    ["DD2_M15091235_M0_LK_SR", "DD2_M15091235_M0_LK_HR"]
#            },
#            "q=1.4":{
#                "DD2_M11461635_M0_LK":
#                     # # 47ms | stable | 3D     # 65ms | stable | 3D (wrong init.data)
#                    ["DD2_M16351146_M0_LK_LR"]  #, "DD2_M11461635_M0_LK_SR"]
#            }
#         },
#     "LS220":
#         {
#            "q=1":{
#                "LS220_M13641364_M0":
#                    # 50ms | BH | 3D          # 41ms | BH | no3D      # 49ms | BH | 3D | wrong BH time
#                    ["LS220_M13641364_M0_SR", "LS220_M13641364_M0_HR", "LS220_M13641364_M0_LR"],
#                "LS220_M13641364_M0_LK":
#                    # 38ms | BH | 3D
#                    ["LS220_M13641364_M0_LK_SR_restart"]
#            },
#            "q=1.1":{
#                "LS220_M14001330_M0":
#                    # 38ms | BH | no3D        # 37ms | BH | no3D
#                    ["LS220_M14001330_M0_HR", "LS220_M14001330_M0_SR"],
#                "LS220_M14351298_M0":
#                    # 38ms | stable | no 3D    # 39ms | BH | no3D
#                    ["LS220_M14351298_M0_HR", "LS220_M14351298_M0_SR"]
#            },
#            "q=1.2":{
#                "LS220_M14691268_M0":
#                     # 49ms | stable | no3D   # 43ms | BH | no3D       # 43ms | stable | np3D
#                    ["LS220_M14691268_M0_SR", "LS220_M14691268_M0_HR", "LS220_M14691268_M0_LR"],
#                "LS220_M14691268_M0_LK":
#                     # 24ms | stable | no3D       # 107ms | long-lived BH | 3D(60ms+)
#                    ["LS220_M14691268_M0_LK_HR", "LS220_M14691268_M0_LK_SR"],
#            },
#            "q=1.4":{
#                "LS220_M16351146_M0_LK":
#                     # 30ms | BH | 3D            # missing 38ms | BH | 3D
#                    ["LS220_M16351146_M0_LK_LR", "LS220_M11461635_M0_LK_SR"]
#            },
#            "q=1.7":{
#                "LS220_M10651772_M0_LK":
#                    # missing 23 ms | BH | PC    # 16ms | BH | 3D | PC
#                    ["LS220_M10651772_M0_LK_SR", "LS220_M10651772_M0_LK_LR"]
#            }
#         },
#     "SFHo":
#         {
#            "q=1":{
#                "SFHo_M13641364_M0":
#                     # 22ms | BH | 3D        # 23ms | BH | 3D
#                    ["SFHo_M13641364_M0_SR", "SFHo_M13641364_M0_HR"],
#                "SFHo_M13641364_M0_LK":
#                     # 29ms | BH | no3D
#                    ["SFHo_M13641364_M0_LK_SR"],
#                "SFHo_M13641364_M0_LK_p2019":
#                     # 24ms | BH | no3D          # 37ms | BH | no3D
#                    ["SFHo_M13641364_M0_LK_HR", "SFHo_M13641364_M0_LK_SR_2019pizza"]
#            },
#            "q=1.1":{
#                "SFHo_M14521283_M0":
#                     # 29ms | BH | 3D        # 32ms | BH | 3D
#                    ["SFHo_M14521283_M0_HR", "SFHo_M14521283_M0_SR"],
#                "SFHo_M14521283_M0_LK_p2019":
#                     # 26ms | BH | no3D         # 26ms | BH | no3D
#                    ["SFHo_M14521283_M0_LK_HR", "SFHo_M14521283_M0_LK_SR_2019pizza"],
#                "SFHo_M14521283_M0_LK_SR":
#                    # 24ms | BH | no3D
#                    ["SFHo_M14521283_M0_LK_SR"]
#            },
#            "q=1.4":{
#                # "SFHo_M11461635_M0_LK": # [ wrong init. data]
#                #      # 65(missing5)ms | stable | 3D
#                #     ["SFHo_M11461635_M0_LK_SR"],
#                "SFHo_M16351146_M0_LK_p2019":
#                     # 31ms | BH | 3D
#                    ["SFHo_M16351146_M0_LK_LR"]
#            },
#            # "q=1.7":{
#            #     "SFHo_M10651772_M0_LK": # [wrong init.data]
#            #          # 21ms | BH | 3D | PC      # 26ms | stable | 3D [might be wrong]
#            #         ["SFHo_M10651772_M0_LK_SR","SFHo_M10651772_M0_LK_LR"]
#            # }
#         },
#     "SLy4":
#         {
#            "q=1":{
#                "SLy4_M13641364_M0_LK":
#                    # 21ms | BH | no3D         # 24ms | BH | no3D
#                   ["SLy4_M13641364_M0_LK_LR", "SLy4_M13641364_M0_LK_SR"],
#                "SLy4_M13641364_M0":
#                   # 36ms | BH | 3D
#                   ["SLy4_M13641364_M0_SR"]
#         },
#            "q=1.1":{
#                "SLy4_M14521283_M0":
#                    # 28ms | BH | extracting 3D # 34ms | BH | 3D
#                    ["SLy4_M14521283_M0_LR", "SLy4_M14521283_M0_SR"] # extracting profiles
#            },
#            # "q=1.4":{
#            #     "SLy4_M11461635_M0_LK": # [wrong init. data]
#            #         # 67ms | stable | 3D [ might be wrong! ]
#            #         ["SLy4_M11461635_M0_LK_SR"]
#            # },
#            "q=1.8":{
#                "SLy4_M10201856_M0_LK":
#                    # 17ms | BH | 3D | PC
#                    ["SLy4_M10201856_M0_LK_SR"]
#            }
#     }
# }



# print("Simulations: ")

# for eos in simulations2.keys():
#     for q in simulations2[eos].keys():
#         for unique in simulations2[eos][q].keys():
#             for sim in simulations2[eos][q][unique]:
#                 print(sim + " "),
# print("\ndone")

# ./analyze.sh BLh_M10651772_M0_LK_SR /data1/numrel/WhiskyTHC/Backup/2018/GW170817/ /data01/numrel/vsevolod.nedora/postprocessed4/

if __name__ == '__main__':
    # print all sims:
    print("simulations")
    for eos in old_simulations.keys():
        for u_sim in old_simulations[eos].keys():
            for sim in old_simulations[eos][u_sim]:
                print(sim),
    print("")

    '''
    for sim in LS220_M140140_LK_LR LS220_M140140_LK LS220_M140140_LK_HR LS220_M135135_LK_LR LS220_M135135_LK LS220_M135135_LK_HR LS220_M150150_LK LS220_M144139_LK LS220_M140120_LK LS220_M145145_LK LS220_M160160_LK LS220_M1251365_LK LS220_M120120_LK LS220_M171171_LK LS220_M135135_M0 LS220_M140120_M0 BHBlp_M140140_LK BHBlp_M140140_LK_HR BHBlp_M150150_LK BHBlp_M150150_LK_HR BHBlp_M135135_M0 BHBlp_M140120_LK BHBlp_M160160_LK BHBlp_M140120_M0 BHBlp_M135135_LK BHBlp_M135135_LK_HR BHBlp_M144139_LK BHBlp_M125125_LK BHBlp_M1251365_LK BHBlp_M130130_LK SFHo_M146146_LK SFHo_M140140_LK SFHo_M144139_LK SFHo_M1251365_LK SFHo_M135135_M0 SFHo_M140120_M0 SFHo_M135135_LK_LR SFHo_M135135_LK SFHo_M135135_LK_HR SFHo_M140120_LK DD2_M140120_LK DD2_M160160_LK DD2_M150150_LK DD2_M150150_LK_HR DD2_M125125_LK DD2_M135135_M0 DD2_M144139_LK DD2_M120120_LK DD2_M130130_LK DD2_M1251365_LK DD2_M135135_LK DD2_M135135_LK_HR DD2_M140140_LK DD2_M140140_LK_HR DD2_M140120_M0; do python outflowed.py -i /data1/numrel/WhiskyTHC/Backup/2017/ -o /data01/numrel/vsevolod.nedora/postprocessed_radice2/ -s $sim -t reshape -p 8 -d 0 && python outflowed.py -i /data1/numrel/WhiskyTHC/Backup/2017/ -o /data01/numrel/vsevolod.nedora/postprocessed_radice2/ -s $sim -t all -d 0 -m geo bern_geoend geo_entropy_above_10 geo_entropy_below_10 --overwrite yes; done '''