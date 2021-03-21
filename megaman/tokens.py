from collections import OrderedDict

# Dictionaries sorting Tokens by hierarchy. Hierarchies are based on game importance and similarity.
NULL_TOKENS = OrderedDict(
    {
        "@": "null",
    }
)

SOLID_TOKENS = OrderedDict(
    {
        "#": "solid",
     }
)

PASSABLE_TOKENS = OrderedDict(
    {
        "-": "air",
        "|": "ladder",
        "B": "breakable",
        "M": "moving platform",
        "~": "water",
    }
)




ENEMY_TOKENS = OrderedDict(
    {    
        "e": "enemy",
    }
)
SPECIAL_TOKENS = OrderedDict(
    {
        "H": "hazard",
        "P": "player",
        "Z": "level orb",
     }
)



TOKEN_DOWNSAMPLING_HIERARCHY = [
    NULL_TOKENS,
    SOLID_TOKENS,
    PASSABLE_TOKENS,
    ENEMY_TOKENS,
    SPECIAL_TOKENS,
]

TOKENS = OrderedDict(
    {**NULL_TOKENS, **SOLID_TOKENS, **PASSABLE_TOKENS, **ENEMY_TOKENS, **SPECIAL_TOKENS}
)

TOKEN_GROUPS = [NULL_TOKENS, SOLID_TOKENS, PASSABLE_TOKENS, ENEMY_TOKENS, SPECIAL_TOKENS]

REPLACE_TOKENS = {}


















##MORE COMPLICATED!!!!
# NULL_TOKENS = OrderedDict(
#     {
#         "@": "null",
#     }
# )

# SOLID_TOKENS = OrderedDict(
#     {
#         "#": "solid",
#         "C": "solid",
#         "A": "solid",
#      }
# )

# PASSABLE_TOKENS = OrderedDict(
#     {
#         "-": "air",
#         "|": "ladder",
#         "B": "breakable",
#         "M": "moving platform",
#         "~": "water",
#         "L": "air",
#         "l": "air",
#         "W": "air",
#         "w": "air",
#         "+": "air",
#         "D": "air",
#         "U": "air",
#         "t": "air",
#         "*": "air",
#     }
# )




# ENEMY_TOKENS = OrderedDict(
#     {    
#         "a": "enemy",
#         "b": "enemy",
#         "<": "enemy",
#         "^": "enemy",
#         "c": "enemy",
#         "d": "enemy",
#         "e": "enemy",
#         "f": "enemy",
#         "g": "enemy",
#         "h": "enemy",
#         "i": "enemy",
#         "j": "enemy",
#         "k": "enemy",
#         "m": "enemy",
#         "n": "enemy",
#         "o": "enemy",
#         "p": "enemy",
#         "q": "enemy",
#         "r": "enemy",

#     }
# )
# SPECIAL_TOKENS = OrderedDict(
#     {
#         "H": "hazard",
#         "P": "player",
        
#         "Z": "level orb",
        
#      }
# )



# TOKEN_DOWNSAMPLING_HIERARCHY = [
#     NULL_TOKENS,
#     SOLID_TOKENS,
#     PASSABLE_TOKENS,
#     ENEMY_TOKENS,
#     SPECIAL_TOKENS,
# ]

# TOKENS = OrderedDict(
#     {**NULL_TOKENS, **SOLID_TOKENS, **PASSABLE_TOKENS, **ENEMY_TOKENS, **SPECIAL_TOKENS}
# )

# TOKEN_GROUPS = [NULL_TOKENS, SOLID_TOKENS, PASSABLE_TOKENS, ENEMY_TOKENS, SPECIAL_TOKENS]

# REPLACE_TOKENS = {}
